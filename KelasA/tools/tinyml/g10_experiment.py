"""Eksperimen G10: stabilitas kepentingan fitur Random Forest + dampak pengurangan fitur.

Isi (semua tanpa kebocoran):
  * >= 20 seed (default 20). Tiap seed: StratifiedGroupKFold(shuffle, seed) menurut group
    (= Coffee_ID); seluruh ulangan satu Coffee_ID selalu di sisi yang sama.
  * Di SETIAP fold, peringkat fitur dihitung HANYA dari data training fold itu:
      - impurity importance  (RF pada training fold)
      - permutation importance (RF pada sebagian grup training, dipermutasi pada grup
        training lain; test fold tidak pernah disentuh)
  * Stabilitas peringkat: korelasi Spearman antar peringkat (antar seed/fold) dan
    frekuensi fitur masuk top-k.
  * Dampak pengurangan fitur: untuk tiap k, ambil top-k dari peringkat training fold,
    latih ulang RF, nilai pada test fold. Hasil = rata-rata fold per seed, lalu mean/std antar seed.

Provenance: --kind synthetic_fixture -> hasil hanya `pipeline_sanity_check` (BUKAN akurasi kopi).
            --kind trained           -> `real_dataset_evaluation` (dataset berlabel Anda).
Tidak ada fitur/kanal yang dihapus oleh skrip ini; ia hanya MENGUKUR.
"""
import argparse
import json
import sys
import warnings

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata

import features_ref as fr
import rf_core as rc
import train_export as te
from rf_core import DataError

MIN_SEEDS = 20


def _spearman_matrix(imp):
    R = np.vstack([rankdata(-v) for v in imp])  # peringkat 1 = terpenting; seri = rata-rata
    return np.corrcoef(R)


def _pair_stats(imp):
    """Spearman antar peringkat. Peringkat KONSTAN (semua skor sama, mis. permutation
    importance 0 untuk semua fitur) tak punya korelasi -> dihitung terpisah, bukan dipalsukan."""
    imp = np.asarray(imp)
    const = np.array([np.ptp(v) == 0 for v in imp])
    ok = imp[~const]
    d = dict(n_rankings=len(imp), n_constant_rankings=int(const.sum()), n_pairs=0,
             spearman_mean=None, spearman_std=None, spearman_min=None)
    if len(ok) >= 2:
        C = _spearman_matrix(ok)
        v = C[np.triu_indices(len(ok), 1)]
        d.update(n_pairs=int(len(v)), spearman_mean=float(v.mean()), spearman_std=float(v.std()),
                 spearman_min=float(v.min()))
    return d


def run_experiment(X, y, groups, seeds, ks, n_splits, params, perm_repeats):
    methods = ["impurity", "permutation"]
    F = X.shape[1]
    imp = {m: [] for m in methods}                  # per (seed, fold)
    imp_seed = {m: [[] for _ in seeds] for m in methods}
    perf = {(m, k): [[] for _ in seeds] for m in methods for k in ks}
    perf_bal = {(m, k): [[] for _ in seeds] for m in methods for k in ks}
    agree = []
    n_agree_undefined = 0
    for si, seed in enumerate(seeds):
        for ftr, fte in rc.group_folds(y, groups, seed, n_splits):
            assert not (set(groups[ftr]) & set(groups[fte])), "kebocoran group"
            fold_imp = {}
            for m in methods:  # peringkat HANYA dari training fold
                v = rc.rank_importance(X[ftr], y[ftr], groups[ftr], m, seed, params, perm_repeats)
                fold_imp[m] = v
                imp[m].append(v)
                imp_seed[m][si].append(v)
            if np.ptp(fold_imp["permutation"]) > 0:
                agree.append(float(np.corrcoef(rankdata(-fold_imp["impurity"]), rankdata(-fold_imp["permutation"]))[0, 1]))
            else:
                n_agree_undefined += 1
            for m in methods:
                for k in ks:
                    cols = np.arange(F) if k >= F else rc.top_k(fold_imp[m], k)
                    r = rc.eval_fold(X, y, ftr, fte, seed, params, cols)
                    perf[(m, k)][si].append(r["accuracy"])
                    perf_bal[(m, k)][si].append(r["balanced_accuracy"])
        print(f"seed {seed} selesai", file=sys.stderr)

    out = dict(n_seeds=len(seeds), seeds=list(map(int, seeds)), n_splits=n_splits, perm_repeats=perm_repeats)
    out["rank_stability"] = {}
    out["topk_frequency"] = {}
    for m in methods:
        A = np.vstack(imp[m])
        st = _pair_stats(A)
        # Spearman antar seed: rata-rata importance fold per seed
        per_seed = np.vstack([np.mean(v, axis=0) for v in imp_seed[m]])
        st["spearman_between_seeds"] = _pair_stats(per_seed)
        out["rank_stability"][m] = st
        order = np.argsort(-A, axis=1, kind="stable")
        freq = {}
        for k in ks:
            if k >= F:
                continue
            hit = np.zeros(F)
            for row in order:
                hit[row[:k]] += 1
            f = hit / len(A)
            top = np.argsort(-f, kind="stable")[:min(k, 15)]
            freq[f"top{k}"] = dict(
                n_features_always_in_topk=int((f == 1.0).sum()),
                n_features_ever_in_topk=int((f > 0).sum()),
                most_frequent=[dict(feature=fr.FEATURE_NAMES[i], index=int(i), frequency=float(f[i])) for i in top])
        out["topk_frequency"][m] = freq
        mean_imp = A.mean(axis=0)
        out.setdefault("mean_importance_top10", {})[m] = [
            dict(feature=fr.FEATURE_NAMES[i], index=int(i), mean=float(mean_imp[i]), std=float(A[:, i].std()))
            for i in np.argsort(-mean_imp, kind="stable")[:10]]
        # info per kanal gas (jumlah importance fitur kanal itu) -- informasi saja, tidak ada kanal dihapus
        ch = []
        for c in range(fr.N_GAS):
            sl = slice(c * fr.FEATURES_PER_GAS, (c + 1) * fr.FEATURES_PER_GAS)
            ch.append(dict(channel=f"gas{c+1}", sensor=fr.GAS_NAMES[c], mean_importance_sum=float(mean_imp[sl].sum())))
        ch.append(dict(channel="env(temp,rh)", sensor="DHT22", mean_importance_sum=float(mean_imp[-2:].sum())))
        out.setdefault("channel_importance", {})[m] = ch
    out["impurity_vs_permutation_spearman"] = dict(
        mean=float(np.mean(agree)) if agree else None, std=float(np.std(agree)) if agree else None,
        n=len(agree), n_undefined_permutation_all_zero=n_agree_undefined)

    out["feature_reduction"] = {}
    for m in methods:
        rows = []
        for k in ks:
            acc = np.array([np.mean(v) for v in perf[(m, k)]])
            bal = np.array([np.mean(v) for v in perf_bal[(m, k)]])
            rows.append(dict(k=int(min(k, F)), accuracy_mean=float(acc.mean()), accuracy_std_over_seeds=float(acc.std()),
                             balanced_accuracy_mean=float(bal.mean()),
                             balanced_accuracy_std_over_seeds=float(bal.std())))
        out["feature_reduction"][m] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--kind", required=True, choices=["trained", "synthetic_fixture"])
    ap.add_argument("--seeds", type=int, default=20, help=f"jumlah seed (>= {MIN_SEEDS})")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=0, help="0 = otomatis (maks 5)")
    ap.add_argument("--top-k", default="5,10,15,20,30,58")
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=rc.DEFAULT_RF["max_depth"])
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--out", default="g10_report.json")
    a = ap.parse_args()
    try:
        if a.seeds < MIN_SEEDS:
            raise DataError(f"--seeds harus >= {MIN_SEEDS} (dapat {a.seeds})")
        sessions, labels, groups = te.load_sessions(a.samples, a.labels)
        if a.kind == "trained" and not te.labels_have_group_column(a.labels):
            raise DataError("--kind trained butuh kolom group/Coffee_ID di labels.csv (cegah kebocoran)")
        X, y_lab, ids, skipped = te.build_matrix(sessions, labels)
        classes = sorted(set(y_lab))
        y = np.array([classes.index(l) for l in y_lab])
        g = np.array([groups[i] for i in ids])
        n_splits = a.n_splits or rc.max_splits(y, g)
        if n_splits < 2:
            raise DataError("tiap kelas butuh >= 2 group untuk GroupKFold")
        ks = [int(k) for k in a.top_k.split(",")]
        params = dict(n_estimators=a.n_estimators, max_depth=a.max_depth)
        seeds = list(range(a.seed_start, a.seed_start + a.seeds))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_experiment(X, y, g, seeds, ks, n_splits, params, a.perm_repeats)
    except DataError as e:
        print(f"ERROR data: {e}", file=sys.stderr)
        sys.exit(2)

    synthetic = a.kind == "synthetic_fixture"
    rep = dict(
        experiment="g10_rf_feature_stability", data_provenance="synthetic_fixture" if synthetic else "real_dataset",
        feature_spec=fr.FEATURE_SPEC, classes=classes, n_sessions=len(ids), n_groups=int(len(set(g))),
        skipped_sessions=skipped, grouping="StratifiedGroupKFold menurut group/Coffee_ID",
        leakage_controls=["split per group", "peringkat fitur dihitung hanya dari training fold",
                          "permutation importance memakai split-dalam pada data training"],
        rf_params={**rc.DEFAULT_RF, **params},
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        claims_allowed=("HANYA verifikasi pipeline. Angka di bawah dari data SINTETIK; bukan akurasi/stabilitas fitur kopi."
                        if synthetic else
                        "Hasil pada dataset berlabel yang diberikan. Sebut 'model G10' hanya jika dataset = kopi G10."),
        pipeline_sanity_check=res if synthetic else None,
        real_dataset_evaluation=None if synthetic else res,
    )
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2)
    r = res["rank_stability"]
    print(json.dumps(dict(provenance=rep["data_provenance"], n_seeds=res["n_seeds"], n_splits=n_splits,
                          spearman_impurity=r["impurity"]["spearman_mean"],
                          spearman_permutation=r["permutation"]["spearman_mean"],
                          const_perm_rankings=r["permutation"]["n_constant_rankings"],
                          impurity_vs_permutation=res["impurity_vs_permutation_spearman"],
                          reduction_impurity=res["feature_reduction"]["impurity"]), indent=1))


if __name__ == "__main__":
    main()
