"""Training offline + ekspor model ke firmware (src/tinyml/model_params.rs).

Skema input (CSV, header wajib):
  samples.csv : measurement_id,t_s,gas1,...,gas8,temperature,humidity
                [,dht_age_s] [,gas1_valid,...,gas8_valid]
                - gas1..gas8 = tegangan (V) dalam urutan payload MQTT
                  (gas1=MQ-6, gas2=MQ-135, gas3=MQ-7, gas4=MQ-3,
                   gas5=TGS2602, gas6=TGS2620, gas7=TGS2611, gas8=TGS2600)
                - sel kosong / nan / inf  => sample tidak valid (tidak diganti nilai)
                - gasN_valid (0/1) opsional; dht_age_s > 10 => suhu/RH tidak valid
  labels.csv  : measurement_id,label[,group]
                - satu baris per sesi; group = unit fisik independen
                  (mis. aliquot/batch). Default group = measurement_id.
                  Split train/test SELALU per group (tanpa kebocoran).
  Kolom group boleh bernama `group` atau `Coffee_ID`. Untuk --kind trained kolom ini
  WAJIB ada (satu Coffee_ID = satu biji/lot fisik; semua ulangannya satu group).

Penggunaan (default --model rf = Random Forest; --model linear = jalur lama):
  python train_export.py --samples S --labels L --kind trained --version v1 \
      --out-rs ../../src/tinyml/model_params.rs --report training_report.json
"""
import argparse
import csv
import json
import math
import re
import sys

import numpy as np

import features_ref as fr
import rf_core as rc
from rf_core import DataError

LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,12}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,12}$")


def _num(s):
    s = (s or "").strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        raise DataError(f"nilai bukan angka: {s!r}")


def labels_have_group_column(labels_path):
    with open(labels_path, newline="") as f:
        cols = set(csv.DictReader(f).fieldnames or [])
    return bool({"group", "Coffee_ID"} & cols)


def load_sessions(samples_path, labels_path):
    with open(labels_path, newline="") as f:
        rd = csv.DictReader(f)
        need = {"measurement_id", "label"}
        if not need <= set(rd.fieldnames or []):
            raise DataError(f"labels.csv butuh kolom {sorted(need)}")
        labels, groups = {}, {}
        for row in rd:
            mid = row["measurement_id"].strip()
            if mid in labels:
                raise DataError(f"measurement_id duplikat di labels: {mid}")
            lab = row["label"].strip()
            if not LABEL_RE.match(lab):
                raise DataError(f"label tidak valid {lab!r} (hanya [A-Za-z0-9_-], max 12)")
            labels[mid] = lab
            groups[mid] = (row.get("group") or row.get("Coffee_ID") or mid).strip()

    gas_cols = [f"gas{i}" for i in range(1, 9)]
    with open(samples_path, newline="") as f:
        rd = csv.DictReader(f)
        cols = set(rd.fieldnames or [])
        need = {"measurement_id", "t_s", "temperature", "humidity", *gas_cols}
        if not need <= cols:
            raise DataError(f"samples.csv kurang kolom: {sorted(need - cols)}")
        valid_cols = [f"gas{i}_valid" for i in range(1, 9)]
        has_valid = all(c in cols for c in valid_cols)
        has_age = "dht_age_s" in cols
        raw = {}
        for row in rd:
            mid = row["measurement_id"].strip()
            gas = [_num(row[c]) for c in gas_cols]
            gv = [bool(int(float(row[c]))) for c in valid_cols] if has_valid else [True] * 8
            temp, rh = _num(row["temperature"]), _num(row["humidity"])
            ev = True
            if has_age:
                age = _num(row["dht_age_s"])
                ev = math.isfinite(age) and age <= fr.DHT_MAX_AGE_S
            raw.setdefault(mid, []).append((_num(row["t_s"]), gas, gv, temp, rh, ev))

    unknown = set(raw) - set(labels)
    if unknown:
        raise DataError(f"sesi tanpa label: {sorted(unknown)[:5]}")
    sessions = {}
    for mid, rows in raw.items():
        ts = [r[0] for r in rows]
        if any(not math.isfinite(t) for t in ts) or any(b < a for a, b in zip(ts, ts[1:])):
            raise DataError(f"t_s tidak finite / tidak naik di sesi {mid}")
        sessions[mid] = dict(
            t=np.array(ts), gas=np.array([r[1] for r in rows]),
            gv=np.array([r[2] for r in rows]), temp=np.array([r[3] for r in rows]),
            rh=np.array([r[4] for r in rows]), ev=np.array([r[5] for r in rows]),
        )
    return sessions, labels, groups


def build_matrix(sessions, labels):
    X, y_lab, ids, skipped = [], [], [], []
    for mid, s in sorted(sessions.items()):
        try:
            X.append(fr.session_features(s["t"], s["gas"], s["gv"], s["temp"], s["rh"], s["ev"]))
            y_lab.append(labels[mid])
            ids.append(mid)
        except fr.FeatureError as e:
            skipped.append((mid, e.kind))
    if not X:
        raise DataError("tidak ada sesi yang memenuhi syarat fitur")
    return np.array(X), y_lab, ids, skipped


def group_split(ids, y_lab, groups, test_frac, seed):
    rng = np.random.default_rng(seed)
    g_of = {i: groups[i] for i in ids}
    cls_of_group = {}
    for i, lab in zip(ids, y_lab):
        if cls_of_group.setdefault(g_of[i], lab) != lab:
            raise DataError(f"group {g_of[i]} berisi lebih dari satu label")
    test_groups = set()
    for lab in sorted(set(y_lab)):
        gs = sorted(g for g, l in cls_of_group.items() if l == lab)
        if len(gs) < 2:
            raise DataError(f"kelas {lab} hanya punya {len(gs)} group; butuh >= 2 untuk split train/test")
        rng.shuffle(gs)
        n_test = min(len(gs) - 1, max(1, int(round(test_frac * len(gs)))))
        test_groups.update(gs[:n_test])
    is_test = np.array([g_of[i] in test_groups for i in ids])
    return ~is_test, is_test


def train_softmax(Z, y, k, l2=1e-2, lr=0.1, iters=3000):
    n, f = Z.shape
    W = np.zeros((k, f))
    b = np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(iters):
        logits = Z @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (G.T @ Z + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b


def softmax_ref(x, mean, std, W, b):
    """Referensi prediksi dari parameter YANG DIEKSPOR (sudah dibulatkan ke f32)."""
    z = (x - mean) / std
    logits = W @ z + b
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(np.argmax(p)), float(p.max())


def _lit(v):
    s = format(float(np.float32(v)), ".9g")
    return s if any(ch in s for ch in ".e") else s + ".0"


def _arr(a):
    return "&[" + ", ".join(_lit(v) for v in np.asarray(a).ravel()) + "]"


def export_rs(path, kind, version, classes, mean, std, W, b, note):
    kind_rs = {"trained": "Trained", "synthetic_fixture": "SyntheticFixture"}[kind]
    with open(path, "w", newline="\n") as f:
        f.write(f"""//! GENERATED oleh tools/tinyml/train_export.py -- JANGAN diedit manual.
//! kind = {kind} | version = {version} | fitur = {fr.FEATURE_SPEC}
//! {note}
//! Normalisasi (mean/std) berasal dari DATA TRAINING, bukan dihitung saat inference.

use super::features::FEATURE_SPEC;
use super::model::{{LinearSoftmaxModel, ModelKind}};

pub static MODEL: LinearSoftmaxModel = LinearSoftmaxModel {{
    kind: ModelKind::{kind_rs},
    version: "{version}",
    feature_spec: FEATURE_SPEC,
    labels: &[{", ".join(f'"{c}"' for c in classes)}],
    weights: {_arr(W)},
    bias: {_arr(b)},
    mean: {_arr(mean)},
    std: {_arr(std)},
}};
""")


def export_rs_forest(path, kind, version, classes, flat, columns_note, note):
    kind_rs = {"trained": "Trained", "synthetic_fixture": "SyntheticFixture"}[kind]
    nodes = []
    for feat, thr, left, right in flat["nodes"]:
        if feat == rc.FOREST_LEAF:
            nodes.append(f"lf({left})")
        else:
            nodes.append(f"nd({feat}, {_lit(thr)}, {left}, {right})")
    node_txt = ",\n    ".join(", ".join(nodes[i:i + 6]) for i in range(0, len(nodes), 6))
    probs = flat["leaf_probs"].ravel()
    k = len(classes)
    prob_txt = ",\n    ".join(", ".join(_lit(v) for v in probs[i:i + 12]) for i in range(0, len(probs), 12))
    st = rc.flat_stats(flat)
    with open(path, "w", newline="\n") as f:
        f.write(f"""//! GENERATED oleh tools/tinyml/train_export.py -- JANGAN diedit manual.
//! model = random_forest | kind = {kind} | version = {version} | fitur = {fr.FEATURE_SPEC}
//! {note}
//! {st['n_trees']} pohon, {st['n_nodes']} node, {st['n_leaves']} daun, kelas = {k}; {columns_note}
//! Fitur MENTAH (tanpa normalisasi). Threshold float32 dibulatkan ke bawah agar
//! `x <= thr` identik dengan scikit-learn.

use super::features::FEATURE_SPEC;
use super::model::{{lf, nd, ForestNode, ModelKind, RandomForestModel}};

static ROOTS: [u16; {st['n_trees']}] = [{", ".join(str(r) for r in flat["roots"])}];

static NODES: [ForestNode; {st['n_nodes']}] = [
    {node_txt}
];

static LEAF_PROBS: [f32; {len(probs)}] = [
    {prob_txt}
];

pub static MODEL: RandomForestModel = RandomForestModel {{
    kind: ModelKind::{kind_rs},
    version: "{version}",
    feature_spec: FEATURE_SPEC,
    labels: &[{", ".join(f'"{c}"' for c in classes)}],
    roots: &ROOTS,
    nodes: &NODES,
    leaf_probs: &LEAF_PROBS,
}};
""")


def export_golden(path, sessions_named, predict_fn, parity_X=None):
    """sessions_named: list of (name, dict(t,gas,gv,temp,rh,ev)).
    predict_fn(x_float32_vector) -> (class_index, score) = referensi dari parameter YANG DIEKSPOR.
    parity_X: matriks fitur (n,58) tambahan untuk uji paritas prediksi Python<->Rust."""
    out = [f"// GENERATED oleh tools/tinyml/train_export.py (golden fixture).\n"
           f"pub const GOLDEN_FEATURE_SPEC: &str = \"{fr.FEATURE_SPEC}\";\n"
           "pub struct GFrame { pub t_s: u32, pub gas: [f32; 8], pub valid: u8, pub temp: f32, pub rh: f32, pub env_ok: bool }\n"
           "pub struct GSession { pub name: &'static str, pub frames: &'static [GFrame],\n"
           "    pub expected_error: &'static str, pub features: &'static [f64],\n"
           "    pub class_index: usize, pub score: f64 }\n"
           "/// Vektor fitur langsung (tanpa frame) untuk paritas prediksi model Python<->Rust.\n"
           "pub struct PVec { pub x: [f32; 58], pub class_index: usize, pub score: f64 }\n"
           "pub static GOLDEN: &[GSession] = &[\n"]
    for name, s in sessions_named:
        frames = []
        for i in range(len(s["t"])):
            mask = sum(1 << c for c in range(8) if s["gv"][i][c])
            gas = ", ".join(_lit(v if math.isfinite(v) else 0.0) for v in s["gas"][i])
            frames.append(f"GFrame {{ t_s: {int(s['t'][i])}, gas: [{gas}], valid: {mask}, "
                          f"temp: {_lit(s['temp'][i])}, rh: {_lit(s['rh'][i])}, env_ok: {str(bool(s['ev'][i])).lower()} }}")
        try:
            x = fr.session_features(s["t"], s["gas"], s["gv"], s["temp"], s["rh"], s["ev"])
            ci, sc = predict_fn(x.astype(np.float32))
            feats = "&[" + ", ".join(repr(float(v)) for v in x) + "]"
            err, ci_s, sc_s = "", ci, repr(sc)
        except fr.FeatureError as e:
            feats, err, ci_s, sc_s = "&[]", e.kind, 0, "0.0"
        out.append(f"GSession {{ name: \"{name}\", expected_error: \"{err}\", features: {feats},\n"
                   f"  class_index: {ci_s}, score: {sc_s},\n  frames: &[\n    " + ",\n    ".join(frames) + ",\n  ] },\n")
    out.append("];\n\npub static PARITY: &[PVec] = &[\n")
    for x in (parity_X if parity_X is not None else []):
        x32 = np.asarray(x, dtype=np.float32)
        ci, sc = predict_fn(x32)
        out.append("  PVec { x: [" + ", ".join(_lit(v) for v in x32) + f"], class_index: {ci}, score: {sc!r} }},\n")
    out.append("];\n")
    with open(path, "w", newline="\n") as f:
        f.write("".join(out))


def _warning(kind):
    return ("DATA SINTETIK -- metrik hanya memverifikasi pipeline, BUKAN akurasi aroma kopi."
            if kind == "synthetic_fixture" else
            "Metrik dari split per-group (Coffee_ID) pada dataset berlabel yang Anda berikan.")


def _provenance(kind):
    return "synthetic_fixture" if kind == "synthetic_fixture" else "real_dataset"


def _wrap_metrics(kind, metrics):
    """Pisahkan hasil fixture sintetis dari evaluasi dataset nyata (kunci berbeda)."""
    if kind == "synthetic_fixture":
        return dict(pipeline_sanity_check=metrics, real_dataset_evaluation=None,
                    claims_allowed="HANYA verifikasi pipeline. Bukan akurasi klasifikasi kopi.")
    return dict(pipeline_sanity_check=None, real_dataset_evaluation=metrics,
                claims_allowed="Boleh dilaporkan sebagai evaluasi dataset ini (split per-group). "
                               "Sebut 'model G10' hanya jika dataset = kopi G10 berlabel.")


def _check_args(kind, version):
    if kind not in ("trained", "synthetic_fixture"):
        raise DataError("kind harus trained atau synthetic_fixture")
    if not VERSION_RE.match(version):
        raise DataError("version hanya [A-Za-z0-9_-], max 12")


def _prep(sessions, labels, groups, groups_explicit, kind):
    if kind == "trained" and not groups_explicit:
        raise DataError("--kind trained butuh kolom group/Coffee_ID di labels.csv "
                        "(tanpa itu ulangan satu biji kopi bisa bocor antara train dan test)")
    X, y_lab, ids, skipped = build_matrix(sessions, labels)
    classes = sorted(set(y_lab))
    if len(classes) < 2:
        raise DataError("butuh >= 2 kelas")
    y = np.array([classes.index(l) for l in y_lab])
    return X, y_lab, y, ids, skipped, classes


def run_linear(sessions, labels, groups, kind, version, out_rs, report_path,
               test_frac=0.3, seed=0, note="", extra_golden=None, golden_out=None,
               groups_explicit=True):
    """JALUR LAMA (softmax linear) -- dipertahankan; butuh model_params versi linear."""
    _check_args(kind, version)
    X, y_lab, y, ids, skipped, classes = _prep(sessions, labels, groups, groups_explicit, kind)
    tr, te = group_split(ids, y_lab, groups, test_frac, seed)
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    const = std < 1e-9
    std = np.where(const, 1.0, std)
    Z = (X - mean) / std
    W, b = train_softmax(Z[tr], y[tr], len(classes))

    mean32, std32 = (mean.astype(np.float32).astype(np.float64), std.astype(np.float32).astype(np.float64))
    W32, b32 = W.astype(np.float32).astype(np.float64), b.astype(np.float32).astype(np.float64)
    pred = np.array([softmax_ref(x, mean32, std32, W32, b32)[0] for x in X])
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y[te], pred[te]):
        cm[t, p] += 1
    metrics = dict(train_accuracy=float((pred[tr] == y[tr]).mean()), test_accuracy=float((pred[te] == y[te]).mean()),
                   test_confusion_matrix=cm.tolist())
    rep = dict(model="linear_softmax", data_provenance=_provenance(kind), kind=kind, version=version,
               feature_spec=fr.FEATURE_SPEC, classes=classes, n_sessions=len(ids), n_groups=len(set(groups[i] for i in ids)),
               n_train=int(tr.sum()), n_test=int(te.sum()), skipped_sessions=skipped,
               constant_features=[fr.FEATURE_NAMES[i] for i in np.where(const)[0]],
               **_wrap_metrics(kind, metrics), warning=_warning(kind))
    export_rs(out_rs, kind, version, classes, mean, std, W, b, note or rep["warning"])
    if report_path:
        with open(report_path, "w") as f:
            json.dump(rep, f, indent=2)
    if golden_out:
        named = [(ids[i], sessions[ids[i]]) for i in np.where(te)[0][:2]] + list(extra_golden or [])
        export_golden(golden_out, named, lambda x: softmax_ref(x.astype(np.float64), mean32, std32, W32, b32), X[:60])
    return rep


def run_rf(sessions, labels, groups, kind, version, out_rs, report_path,
           test_frac=0.3, seed=0, note="", extra_golden=None, golden_out=None,
           groups_explicit=True, params=None, select_top_k=0, select_method="impurity", cv_splits=0):
    """Random Forest: split per group; pemilihan fitur (opsional) HANYA dari data training."""
    _check_args(kind, version)
    X, y_lab, y, ids, skipped, classes = _prep(sessions, labels, groups, groups_explicit, kind)
    g = np.array([groups[i] for i in ids])
    tr, te = group_split(ids, y_lab, groups, test_frac, seed)
    assert not (set(g[tr]) & set(g[te])), "kebocoran group train/test"

    columns = np.arange(fr.FEATURE_COUNT)
    sel_note = "semua 58 fitur tersedia bagi pohon"
    if select_top_k:
        imp = rc.rank_importance(X[tr], y[tr], g[tr], select_method, seed, params)  # TRAINING saja
        columns = rc.top_k(imp, select_top_k)
        sel_note = f"fitur dibatasi top-{select_top_k} ({select_method}, dari data training saja)"
    rf = rc.fit_sk(X[tr], y[tr], seed, params, columns)
    flat = rc.flatten_forest(rf, columns, len(classes))
    n_par = _parity(rf, flat, X, columns)  # struktur yang DIEKSPOR vs sklearn

    pred = np.array([rc.flat_predict(flat, x)[0] for x in X])
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y[te], pred[te]):
        cm[t, p] += 1
    holdout = dict(train_accuracy=float((pred[tr] == y[tr]).mean()), test_accuracy=float((pred[te] == y[te]).mean()),
                   test_balanced_accuracy=float(rc.balanced_accuracy_score(y[te], pred[te])),
                   test_confusion_matrix=cm.tolist(), n_test_groups=int(len(set(g[te]))))

    cv = None
    n_splits = cv_splits or rc.max_splits(y, g)
    if n_splits >= 2:
        accs, bals = [], []
        for ftr, fte in rc.group_folds(y, g, seed, n_splits):
            cols = np.arange(fr.FEATURE_COUNT)
            if select_top_k:  # seleksi ulang di DALAM fold, hanya dari training fold
                cols = rc.top_k(rc.rank_importance(X[ftr], y[ftr], g[ftr], select_method, seed, params), select_top_k)
            m = rc.eval_fold(X, y, ftr, fte, seed, params, cols)
            accs.append(m["accuracy"]); bals.append(m["balanced_accuracy"])
        cv = dict(n_splits=n_splits, folding="StratifiedGroupKFold(group=Coffee_ID)",
                  accuracy_mean=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
                  balanced_accuracy_mean=float(np.mean(bals)), balanced_accuracy_std=float(np.std(bals)),
                  selection_inside_fold=bool(select_top_k))
    holdout["group_cv"] = cv

    stats = rc.flat_stats(flat)
    rep = dict(model="random_forest", data_provenance=_provenance(kind), kind=kind, version=version,
               feature_spec=fr.FEATURE_SPEC, classes=classes, n_sessions=len(ids), n_groups=int(len(set(g))),
               n_train=int(tr.sum()), n_test=int(te.sum()), skipped_sessions=skipped, seed=seed,
               rf_params={**rc.DEFAULT_RF, **(params or {})}, feature_selection=sel_note,
               selected_features=[fr.FEATURE_NAMES[i] for i in columns] if select_top_k else "all",
               export_vs_sklearn_parity_vectors=n_par, model_size=stats,
               **_wrap_metrics(kind, holdout), warning=_warning(kind),
               esp32_measurements="TIDAK diukur di sini (tanpa board/toolchain esp). flash_bytes = ukuran artefak model; "
                                  "RAM/waktu inferensi ESP32 harus diukur di perangkat (lihat README).")
    export_rs_forest(out_rs, kind, version, classes, flat, sel_note, note or rep["warning"])
    if report_path:
        with open(report_path, "w") as f:
            json.dump(rep, f, indent=2)
    if golden_out:
        rng = np.random.default_rng(seed + 99)
        jit = [X[rng.integers(len(X))] * (1 + rng.normal(0, 0.08, X.shape[1])) + rng.normal(0, 0.01, X.shape[1])
               for _ in range(120)]
        pX = np.vstack([X, np.array(jit)])
        named = [(ids[i], sessions[ids[i]]) for i in np.where(te)[0][:2]] + list(extra_golden or [])
        export_golden(golden_out, named, lambda x: rc.flat_predict(flat, x), pX)
    return rep


def _parity(rf, flat, X, columns):
    """Bandingkan struktur ekspor (indeks fitur ASLI 0..57) dengan sklearn (kolom terpilih)."""
    Xs = rc.f32(X)[:, columns]
    P = rf.predict_proba(Xs)
    bad = 0
    for x, p in zip(rc.f32(X), P):
        c, s = rc.flat_predict(flat, x)
        srt = np.sort(p)
        if abs(s - float(p.max())) > 1e-5 or (c != int(np.argmax(p)) and (srt[-1] - srt[-2]) >= 1e-5):
            bad += 1
    if bad:
        raise DataError(f"parity ekspor-vs-sklearn GAGAL pada {bad}/{len(X)} vektor")
    return int(len(X))


def run(sessions, labels, groups, kind, version, out_rs, report_path, model="rf", **kw):
    if model == "rf":
        return run_rf(sessions, labels, groups, kind, version, out_rs, report_path, **kw)
    if model == "linear":
        for k in ("params", "select_top_k", "select_method", "cv_splits"):
            kw.pop(k, None)
        return run_linear(sessions, labels, groups, kind, version, out_rs, report_path, **kw)
    raise DataError("model harus rf atau linear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--kind", required=True, choices=["trained", "synthetic_fixture"])
    ap.add_argument("--version", required=True)
    ap.add_argument("--out-rs", required=True)
    ap.add_argument("--model", default="rf", choices=["rf", "linear"])
    ap.add_argument("--report", default="training_report.json")
    ap.add_argument("--golden-out")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=rc.DEFAULT_RF["n_estimators"])
    ap.add_argument("--max-depth", type=int, default=rc.DEFAULT_RF["max_depth"])
    ap.add_argument("--min-samples-leaf", type=int, default=rc.DEFAULT_RF["min_samples_leaf"])
    ap.add_argument("--select-top-k", type=int, default=0, help="0 = tanpa seleksi fitur (default)")
    ap.add_argument("--select-method", default="impurity", choices=["impurity", "permutation"])
    a = ap.parse_args()
    try:
        sessions, labels, groups = load_sessions(a.samples, a.labels)
        kw = dict(test_frac=a.test_frac, seed=a.seed, golden_out=a.golden_out,
                  groups_explicit=labels_have_group_column(a.labels))
        if a.model == "rf":
            kw.update(params=dict(n_estimators=a.n_estimators, max_depth=a.max_depth,
                                  min_samples_leaf=a.min_samples_leaf),
                      select_top_k=a.select_top_k, select_method=a.select_method)
        rep = run(sessions, labels, groups, a.kind, a.version, a.out_rs, a.report, model=a.model, **kw)
    except DataError as e:
        print(f"ERROR data: {e}", file=sys.stderr)
        sys.exit(2)
    ev = rep["real_dataset_evaluation"] or rep["pipeline_sanity_check"]
    print(json.dumps(dict(model=rep["model"], data_provenance=rep["data_provenance"], n_train=rep["n_train"],
                          n_test=rep["n_test"], metrics=ev, warning=rep["warning"]), indent=2))


if __name__ == "__main__":
    main()
