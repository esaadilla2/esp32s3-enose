"""Membuat dataset SINTETIK + model fixture + golden test.

DATA INI PALSU (dibangkitkan dari RNG). Dipakai HANYA untuk menguji pipeline.
Model hasilnya bertanda kind=synthetic_fixture dan TIDAK valid untuk aroma kopi.
"""
import argparse
import os

import numpy as np

import features_ref as fr
import train_export as te


def make_session(rng, amp, n=120, base=None):
    t = np.arange(n, dtype=np.float64)
    if base is None:
        base = rng.uniform(0.8, 1.6, size=8)
    tau = rng.uniform(15, 40, size=8)
    gas = base + amp * (1 - np.exp(-t[:, None] / tau)) + rng.normal(0, 0.01, size=(n, 8))
    temp = 25 + rng.normal(0, 0.2, size=n)
    rh = 60 + rng.normal(0, 0.5, size=n)
    return dict(t=t, gas=gas, gv=np.ones((n, 8), bool), temp=temp, rh=rh, ev=np.ones(n, bool))


def write_csv(dirpath, sessions, labels, groups):
    """Contoh skema CSV yang dibaca train_export.py (data SINTETIK)."""
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "samples.csv"), "w", newline="") as f:
        f.write("measurement_id,t_s," + ",".join(f"gas{i}" for i in range(1, 9)) + ",temperature,humidity\n")
        for mid, s in sessions.items():
            for i in range(len(s["t"])):
                f.write(f"{mid},{int(s['t'][i])}," + ",".join(f"{v:.6f}" for v in s["gas"][i])
                        + f",{s['temp'][i]:.4f},{s['rh'][i]:.4f}\n")
    with open(os.path.join(dirpath, "labels.csv"), "w", newline="") as f:
        f.write("measurement_id,label,group\n")
        for mid in sessions:
            f.write(f"{mid},{labels[mid]},{groups[mid]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", help="(opsional) tulis contoh samples.csv/labels.csv sintetis")
    ap.add_argument("--out-rs", required=True)
    ap.add_argument("--golden-out", required=True)
    ap.add_argument("--report", default="synthetic_training_report.json")
    ap.add_argument("--model", default="rf", choices=["rf", "linear"])
    a = ap.parse_args()

    rng = np.random.default_rng(1234)
    amps = {"synth_a": rng.uniform(0.0, 0.2, 8), "synth_b": rng.uniform(0.3, 0.6, 8), "synth_c": rng.uniform(-0.3, 0.0, 8)}
    sessions, labels, groups = {}, {}, {}
    # 8 "biji" sintetis per kelas (group, meniru Coffee_ID), 3 ulangan per biji.
    # Tiap biji punya baseline sendiri; ulangan satu biji TIDAK boleh terpisah train/test.
    for lab, amp in amps.items():
        for gi in range(8):
            gid = f"SYNCOF_{lab}_{gi:02d}"
            base = rng.uniform(0.8, 1.6, size=8)
            gamp = amp + rng.normal(0, 0.12, 8)  # variasi antar-biji: kelas saling tumpang-tindih
            for rep_i in range(3):
                mid = f"SYN_{lab}_{gi:02d}_r{rep_i}"
                sessions[mid] = make_session(rng, gamp, base=base * (1 + rng.normal(0, 0.01, 8)))
                labels[mid], groups[mid] = lab, gid

    # Kasus tepi untuk golden (tidak ikut training):
    e1 = make_session(rng, amps["synth_a"])           # 10% dropout di gas3
    drop = rng.random(len(e1["t"])) < 0.10
    e1["gv"][drop, 2] = False
    e2 = make_session(rng, amps["synth_b"])           # gas5 mati setelah t=30 -> valid < 60
    e2["gv"][30:, 4] = False
    e3 = make_session(rng, amps["synth_c"], n=40)     # terlalu pendek
    extra = [("edge_dropout_gas3", e1), ("edge_gas5_dead", e2), ("edge_too_short", e3)]

    if a.csv_dir:
        write_csv(a.csv_dir, sessions, labels, groups)
    rep = te.run(sessions, labels, groups, "synthetic_fixture", "synth-fx-1", a.out_rs, a.report,
                 model=a.model, note="DATA SINTETIK -- hanya uji pipeline, BUKAN model aroma kopi.",
                 extra_golden=extra, golden_out=a.golden_out)
    print({k: rep[k] for k in ("model", "data_provenance", "n_train", "n_test", "pipeline_sanity_check", "warning")})


if __name__ == "__main__":
    main()
