"""Inti Random Forest (training offline) untuk pipeline TinyML kopi.

Dipakai oleh train_export.py (training + ekspor) dan g10_experiment.py
(eksperimen 20 seed). Hanya alat HOST: butuh numpy + scikit-learn (+ scipy untuk
eksperimen). Firmware Rust TIDAK memakai library apa pun -- ia mengeksekusi
struktur pohon hasil ekspor (`flatten_forest`) dengan aturan yang SAMA
dengan `flat_predict` di bawah.

Aturan konsistensi Python <-> Rust:
  * fitur MENTAH (tanpa z-score); pohon tidak berubah thd penskalaan monoton
  * sklearn membandingkan X(float32) <= threshold(float64). Firmware memakai
    threshold float32, maka threshold DIBULATKAN KE BAWAH ke float32 terdekat
    (`round_down_f32`) supaya  x <= thr64  <=>  x <= thr32  untuk semua x float32.
  * prediksi = rata-rata distribusi kelas daun semua pohon (soft voting =
    predict_proba sklearn); kelas = argmax (seri -> indeks terkecil);
    `score` = nilai rata-rata itu (fraksi suara), BUKAN probabilitas terkalibrasi.
"""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

FOREST_LEAF = 0xFF
MAX_NODES = 60_000      # = model.rs MAX_NODES (indeks node u16)
MAX_TREES = 256         # = model.rs MAX_TREES
MAX_CLASSES = 16        # = model.rs MAX_CLASSES

DEFAULT_RF = dict(n_estimators=50, max_depth=8, min_samples_leaf=2,
                  max_features="sqrt", bootstrap=True)


class DataError(Exception):
    pass


def f32(a):
    return np.asarray(a, dtype=np.float32)


def round_down_f32(thr):
    t = np.float32(thr)
    if float(t) > float(thr):
        t = np.nextafter(t, np.float32(-np.inf))
    return t


def fit_sk(X, y, seed, params=None, columns=None):
    """RandomForestClassifier pada X[:, columns] (X dicast ke float32 seperti di firmware)."""
    p = dict(DEFAULT_RF)
    p.update(params or {})
    cols = np.arange(X.shape[1]) if columns is None else np.asarray(columns)
    rf = RandomForestClassifier(random_state=int(seed), n_jobs=1, **p)
    rf.fit(f32(X)[:, cols], y)
    return rf


def predict_sk(rf, X, columns=None):
    cols = np.arange(X.shape[1]) if columns is None else np.asarray(columns)
    return rf.predict_proba(f32(X)[:, cols])


def flatten_forest(rf, columns, n_classes):
    """-> dict(roots, nodes[(feat, thr32, left, right)], leaf_probs (n_leaf, K) float32)."""
    if len(rf.classes_) != n_classes:
        raise DataError("training tidak memuat semua kelas; model tidak bisa diekspor")
    if not (1 <= len(rf.estimators_) <= MAX_TREES):
        raise DataError(f"jumlah pohon harus 1..{MAX_TREES}")
    columns = [int(c) for c in columns]
    roots, nodes, leaves = [], [], []
    for est in rf.estimators_:
        t = est.tree_
        off = len(nodes)
        roots.append(off)
        for i in range(t.node_count):
            l, r = int(t.children_left[i]), int(t.children_right[i])
            if l == -1:
                v = t.value[i, 0, :].astype(np.float64)
                nodes.append((FOREST_LEAF, np.float32(0.0), len(leaves), 0))
                leaves.append((v / v.sum()).astype(np.float32))
            else:
                if not (l > i and r > i):
                    raise DataError("urutan node sklearn tidak sesuai asumsi (anak > induk)")
                nodes.append((columns[int(t.feature[i])], round_down_f32(t.threshold[i]), off + l, off + r))
    if len(nodes) > MAX_NODES:
        raise DataError(f"{len(nodes)} node > batas {MAX_NODES}; kecilkan n_estimators/max_depth")
    return dict(roots=roots, nodes=nodes, leaf_probs=np.vstack(leaves), n_classes=n_classes)


def flat_predict(flat, x):
    """Meniru persis predict_forest di Rust (float32, urutan akumulasi sama)."""
    x = f32(x)
    nodes, k = flat["nodes"], flat["n_classes"]
    acc = np.zeros(k, dtype=np.float32)
    for root in flat["roots"]:
        i = root
        while nodes[i][0] != FOREST_LEAF:
            feat, thr, left, right = nodes[i]
            i = left if x[feat] <= thr else right
        acc = acc + flat["leaf_probs"][nodes[i][2]]
    inv = np.float32(1.0) / np.float32(len(flat["roots"]))
    best = 0
    for c in range(1, k):
        if acc[c] > acc[best]:
            best = c
    return best, float(acc[best] * inv)


def assert_export_matches_sklearn(rf, flat, X, columns=None, tol=1e-5):
    """Struktur yang DIEKSPOR harus memberi hasil sama dengan sklearn (kecuali seri hampir-sama)."""
    P = predict_sk(rf, X, columns)  # sklearn mengonversi ke float32 juga
    bad = 0
    for x, p in zip(f32(X), P):
        c, s = flat_predict(flat, x)
        srt = np.sort(p)
        near_tie = len(srt) > 1 and (srt[-1] - srt[-2]) < 1e-5
        if abs(s - float(p.max())) > tol or (c != int(np.argmax(p)) and not near_tie):
            bad += 1
    if bad:
        raise DataError(f"parity ekspor-vs-sklearn GAGAL pada {bad}/{len(X)} vektor")
    return len(X)


def flat_stats(flat, n_features=58):
    """Ukuran artefak (flash) + batas kerja terburuk. Tidak mengukur waktu ESP32."""
    nodes = flat["nodes"]
    depth = [0] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):  # anak > induk -> hitung dari belakang
        f, _, l, r = nodes[i]
        depth[i] = 0 if f == FOREST_LEAF else 1 + max(depth[l], depth[r])
    per_tree = [depth[r] for r in flat["roots"]]
    return dict(
        n_trees=len(flat["roots"]), n_nodes=len(nodes), n_leaves=int(flat["leaf_probs"].shape[0]),
        flash_bytes=len(nodes) * 12 + int(flat["leaf_probs"].size) * 4 + len(flat["roots"]) * 2,
        max_depth=max(per_tree), worst_case_node_visits=int(sum(d + 1 for d in per_tree)),
        features_used=sorted({n[0] for n in nodes if n[0] != FOREST_LEAF}),
    )


# ------------------------------------------------------------------ peringkat fitur
def _inner_group_split(y, groups, seed):
    sgk = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(seed))
    try:
        return next(iter(sgk.split(np.zeros(len(y)), y, groups)))
    except ValueError as e:
        raise DataError(f"split dalam (permutation importance) gagal: {e}")


def rank_importance(X, y, groups, method, seed, params=None, perm_repeats=5):
    """Skor kepentingan fitur dari DATA YANG DIBERIKAN SAJA (panggil dengan data training).

    impurity    : feature_importances_ RF pada seluruh data training
    permutation : RF dilatih pada sebagian grup training, permutasi pada grup training
                  lain (held-out di dalam training) -- bukan pada test fold.
    """
    if method == "impurity":
        return fit_sk(X, y, seed, params).feature_importances_.astype(np.float64)
    if method == "permutation":
        tr, va = _inner_group_split(y, np.asarray(groups), seed)
        rf = fit_sk(X[tr], y[tr], seed, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = permutation_importance(rf, f32(X[va]), y[va], n_repeats=perm_repeats,
                                       random_state=int(seed), scoring="balanced_accuracy", n_jobs=1)
        return r.importances_mean.astype(np.float64)
    raise DataError(f"metode importance tidak dikenal: {method}")


def top_k(importance, k):
    """Indeks k fitur teratas (urut naik); seri dipecah deterministik (indeks kecil dulu)."""
    order = np.argsort(-np.asarray(importance), kind="stable")[:k]
    return np.sort(order)


def group_folds(y, groups, seed, n_splits):
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    try:
        return list(sgk.split(np.zeros(len(y)), y, groups))
    except ValueError as e:
        raise DataError(f"StratifiedGroupKFold gagal: {e}")


def max_splits(y, groups):
    """n_splits terbesar (<=5) yang masih memberi >=1 grup per kelas di tiap test fold."""
    per_class = {}
    for lab, g in zip(y, groups):
        per_class.setdefault(int(lab), set()).add(g)
    return int(min(5, min(len(v) for v in per_class.values())))


def eval_fold(X, y, tr, te, seed, params, columns=None):
    rf = fit_sk(X[tr], y[tr], seed, params, columns)
    pred = np.array(rf.classes_)[np.argmax(predict_sk(rf, X[te], columns), axis=1)]
    return dict(accuracy=float((pred == y[te]).mean()),
                balanced_accuracy=float(balanced_accuracy_score(y[te], pred)))
