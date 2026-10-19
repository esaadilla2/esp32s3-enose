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
Penggunaan:
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

LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,12}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,12}$")


class DataError(Exception):
    pass


def _num(s):
    s = (s or "").strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        raise DataError(f"nilai bukan angka: {s!r}")


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
            groups[mid] = (row.get("group") or mid).strip()

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


def export_golden(path, sessions_named, model):
    """sessions_named: list of (name, dict(t,gas,gv,temp,rh,ev)). Menulis fixture host-test."""
    mean, std, W, b, classes = model
    mean32, std32 = mean.astype(np.float32).astype(np.float64), std.astype(np.float32).astype(np.float64)
    W32, b32 = W.astype(np.float32).astype(np.float64), b.astype(np.float32).astype(np.float64)
    out = [f"// GENERATED oleh tools/tinyml/train_export.py (golden fixture).\n"
           f"pub const GOLDEN_FEATURE_SPEC: &str = \"{fr.FEATURE_SPEC}\";\n"
           "pub struct GFrame { pub t_s: u32, pub gas: [f32; 8], pub valid: u8, pub temp: f32, pub rh: f32, pub env_ok: bool }\n"
           "pub struct GSession { pub name: &'static str, pub frames: &'static [GFrame],\n"
           "    pub expected_error: &'static str, pub features: &'static [f64],\n"
           "    pub class_index: usize, pub score: f64 }\n"
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
            ci, sc = softmax_ref(x, mean32, std32, W32, b32)
            feats = "&[" + ", ".join(repr(float(v)) for v in x) + "]"
            err, ci_s, sc_s = "", ci, repr(sc)
        except fr.FeatureError as e:
            feats, err, ci_s, sc_s = "&[]", e.kind, 0, "0.0"
        out.append(f"GSession {{ name: \"{name}\", expected_error: \"{err}\", features: {feats},\n"
                   f"  class_index: {ci_s}, score: {sc_s},\n  frames: &[\n    " + ",\n    ".join(frames) + ",\n  ] },\n")
    out.append("];\n")
    with open(path, "w", newline="\n") as f:
        f.write("".join(out))


def run(sessions, labels, groups, kind, version, out_rs, report_path,
        test_frac=0.3, seed=0, note="", extra_golden=None, golden_out=None):
    if kind not in ("trained", "synthetic_fixture"):
        raise DataError("kind harus trained atau synthetic_fixture")
    if not VERSION_RE.match(version):
        raise DataError("version hanya [A-Za-z0-9_-], max 12")
    X, y_lab, ids, skipped = build_matrix(sessions, labels)
    classes = sorted(set(y_lab))
    if len(classes) < 2:
        raise DataError("butuh >= 2 kelas")
    y = np.array([classes.index(l) for l in y_lab])
    tr, te = group_split(ids, y_lab, groups, test_frac, seed)
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    const = std < 1e-9
    std = np.where(const, 1.0, std)  # fitur konstan di training: z=(x-mean)/1
    Z = (X - mean) / std
    W, b = train_softmax(Z[tr], y[tr], len(classes))

    mean32, std32 = (mean.astype(np.float32).astype(np.float64), std.astype(np.float32).astype(np.float64))
    W32, b32 = W.astype(np.float32).astype(np.float64), b.astype(np.float32).astype(np.float64)
    pred = np.array([softmax_ref(x, mean32, std32, W32, b32)[0] for x in X])
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y[te], pred[te]):
        cm[t, p] += 1
    rep = dict(
        kind=kind, version=version, feature_spec=fr.FEATURE_SPEC, classes=classes,
        n_sessions=len(ids), n_train=int(tr.sum()), n_test=int(te.sum()),
        skipped_sessions=skipped, constant_features=[fr.FEATURE_NAMES[i] for i in np.where(const)[0]],
        train_accuracy=float((pred[tr] == y[tr]).mean()), test_accuracy=float((pred[te] == y[te]).mean()),
        test_confusion_matrix=cm.tolist(),
        warning=("DATA SINTETIK -- metrik hanya memverifikasi pipeline, BUKAN akurasi aroma kopi."
                 if kind == "synthetic_fixture" else "Metrik dari split per-group pada dataset yang Anda berikan."),
    )
    export_rs(out_rs, kind, version, classes, mean, std, W, b, note or rep["warning"])
    if report_path:
        with open(report_path, "w") as f:
            json.dump(rep, f, indent=2)
    if golden_out:
        named = [(ids[i], sessions[ids[i]]) for i in np.where(te)[0][:2]] + list(extra_golden or [])
        export_golden(golden_out, named, (mean, std, W, b, classes))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--kind", required=True, choices=["trained", "synthetic_fixture"])
    ap.add_argument("--version", required=True)
    ap.add_argument("--out-rs", required=True)
    ap.add_argument("--report", default="training_report.json")
    ap.add_argument("--golden-out")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    try:
        sessions, labels, groups = load_sessions(a.samples, a.labels)
        rep = run(sessions, labels, groups, a.kind, a.version, a.out_rs, a.report,
                  a.test_frac, a.seed, golden_out=a.golden_out)
    except DataError as e:
        print(f"ERROR data: {e}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({k: rep[k] for k in ("kind", "n_train", "n_test", "test_accuracy", "warning")}, indent=2))


if __name__ == "__main__":
    main()
