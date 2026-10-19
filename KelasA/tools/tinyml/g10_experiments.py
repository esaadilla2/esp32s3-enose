#!/usr/bin/env python3
"""Eksperimen G10: Random Forest, stabilitas ranking fitur, sensor/feature reduction.

Input CSV fitur sesi: satu baris per measurement/session, kolom label dan group
(Coffee_ID/unit independen), sisanya fitur numerik. Jangan berikan data per-frame
sebagai baris independen jika frame berasal dari sesi yang sama.

Contoh:
  python g10_experiments.py --data session_features.csv --label label \
    --group Coffee_ID --out-dir results --seeds 20

Semua preprocessing/seleksi fitur dikerjakan hanya pada train split setiap fold.
Hasil ini untuk eksperimen offline; belum otomatis diekspor ke firmware Rust.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV: satu baris per sesi, fitur numerik + label + group')
    ap.add_argument('--label', default='label')
    ap.add_argument('--group', default='Coffee_ID')
    ap.add_argument('--out-dir', default='g10_results')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--trees', type=int, default=200)
    args = ap.parse_args()
    if args.seeds < 20:
        ap.error('--seeds minimal 20 sesuai instruksi G10')
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    if args.label not in df or args.group not in df:
        ap.error(f'CSV harus memiliki kolom {args.label!r} dan {args.group!r}')
    feature_cols = [c for c in df.columns if c not in (args.label, args.group)]
    if not feature_cols: ap.error('Tidak ada kolom fitur')
    # Missing/non-numeric data is not silently imputed: fail early so data quality is explicit.
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        ap.error('Fitur mengandung kosong/NaN/inf; perbaiki atau dokumentasikan strategi imputasi dahulu.')
    y = df[args.label].astype(str).to_numpy(); groups = df[args.group].astype(str).to_numpy()
    if len(np.unique(y)) < 2: ap.error('Minimal dua kelas dibutuhkan')
    if len(np.unique(groups)) < args.folds: ap.error('Jumlah group independen lebih kecil daripada --folds')
    # Stratified group folds if supported by data; this prevents same Coffee_ID crossing folds.
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    seed_rows, imp_rows, pred_rows = [], [], []
    for seed in range(args.seeds):
        fold_scores=[]
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            if set(groups[tr]) & set(groups[te]): raise RuntimeError('Group leakage detected')
            model = RandomForestClassifier(n_estimators=args.trees, random_state=seed,
                class_weight='balanced', n_jobs=-1, min_samples_leaf=1)
            model.fit(X.iloc[tr], y[tr])
            yp = model.predict(X.iloc[te])
            fold_scores.append({'seed':seed,'fold':fold,'accuracy':accuracy_score(y[te],yp),
                'balanced_accuracy':balanced_accuracy_score(y[te],yp),
                'macro_f1':f1_score(y[te],yp,average='macro',zero_division=0),
                'n_train':int(len(tr)),'n_test':int(len(te))})
            # Both importance methods evaluated on held-out fold; ranking does not use test labels for training.
            perm = permutation_importance(model, X.iloc[te], y[te], n_repeats=10,
                random_state=seed, scoring='balanced_accuracy', n_jobs=-1)
            for i, name in enumerate(feature_cols):
                imp_rows.append({'seed':seed,'fold':fold,'feature':name,
                    'impurity':float(model.feature_importances_[i]),
                    'permutation_mean':float(perm.importances_mean[i]),
                    'permutation_std':float(perm.importances_std[i])})
        seed_rows.extend(fold_scores)
    scores=pd.DataFrame(seed_rows); imp=pd.DataFrame(imp_rows)
    scores.to_csv(out/'cv_scores_by_seed_fold.csv',index=False)
    imp.to_csv(out/'importance_by_seed_fold.csv',index=False)
    # Aggregate rankings and stability metrics over all seed/fold estimates.
    agg=imp.groupby('feature')[['impurity','permutation_mean']].mean()
    agg['impurity_rank']=agg['impurity'].rank(ascending=False,method='average')
    agg['permutation_rank']=agg['permutation_mean'].rank(ascending=False,method='average')
    agg['topk_impurity_frequency']=imp.assign(rank=imp.groupby(['seed','fold'])['impurity'].rank(ascending=False,method='first')).groupby('feature')['rank'].apply(lambda s: float((s<=args.top_k).mean()))
    agg['topk_permutation_frequency']=imp.assign(rank=imp.groupby(['seed','fold'])['permutation_mean'].rank(ascending=False,method='first')).groupby('feature')['rank'].apply(lambda s: float((s<=args.top_k).mean()))
    agg.sort_values('permutation_rank').to_csv(out/'feature_ranking_stability.csv')
    # Spearman/Kendall agreement between mean rankings (descriptive, not an independent accuracy metric).
    sp=spearmanr(agg['impurity_rank'],agg['permutation_rank'])
    kt=kendalltau(agg['impurity_rank'],agg['permutation_rank'])
    report={'purpose':'G10 Random Forest / Feature Selection / Sensor Reduction',
      'n_sessions':int(len(df)),'n_groups':int(len(np.unique(groups))),
      'n_features':len(feature_cols),'n_classes':int(len(np.unique(y))),
      'seeds':args.seeds,'folds':args.folds,'trees_per_forest':args.trees,'top_k':args.top_k,
      'metrics_mean':{c:float(scores[c].mean()) for c in ['accuracy','balanced_accuracy','macro_f1']},
      'metrics_std_across_seed_folds':{c:float(scores[c].std(ddof=1)) for c in ['accuracy','balanced_accuracy','macro_f1']},
      'importance_rank_agreement':{'spearman_rho':float(sp.statistic),'spearman_pvalue':float(sp.pvalue),
                                   'kendall_tau':float(kt.statistic),'kendall_pvalue':float(kt.pvalue)},
      'warning':'Ini evaluasi offline. Pastikan setiap group adalah unit sampel independen. Ranking fitur bukan bukti sensor boleh dihapus; evaluasi ulang model dengan subset fitur di CV tersarang sebelum menyimpulkan reduksi sensor.'}
    (out/'g10_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
