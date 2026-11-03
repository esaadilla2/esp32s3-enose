"""Uji sisi Python (jalankan: python -m unittest test_rf_pipeline -v). Data SINTETIK."""
import os
import tempfile
import unittest

import numpy as np

import features_ref as fr
import rf_core as rc
import make_synthetic_fixture as ms
import train_export as te


def _data():
    rng = np.random.default_rng(5)
    amps = {"a": rng.uniform(0, .2, 8), "b": rng.uniform(.3, .6, 8)}
    sessions, labels, groups = {}, {}, {}
    for lab, amp in amps.items():
        for gi in range(6):
            base = rng.uniform(.8, 1.6, 8)
            for r in range(3):
                mid = f"{lab}{gi}_{r}"
                sessions[mid] = ms.make_session(rng, amp + rng.normal(0, .1, 8), base=base)
                labels[mid], groups[mid] = lab, f"{lab}{gi}"
    return sessions, labels, groups


class T(unittest.TestCase):
    def test_threshold_round_down_equivalent_to_float64_compare(self):
        rng = np.random.default_rng(1)
        for _ in range(20000):
            a = np.float32(rng.normal() * 10.0 ** float(rng.integers(-3, 3)))
            b = np.nextafter(a, np.float32(np.inf))
            thr64 = (float(a) + float(b)) / 2  # titik tengah seperti sklearn
            t32 = rc.round_down_f32(thr64)
            self.assertLessEqual(float(t32), thr64)
            for x in (a, b):
                self.assertEqual(bool(x <= t32), bool(float(x) <= thr64))

    def test_export_matches_sklearn_and_rust_semantics(self):
        s, l, g = _data()
        X, yl, ids, _ = te.build_matrix(s, l)
        y = np.array([0 if v == "a" else 1 for v in yl])
        cols = np.arange(58)
        rf = rc.fit_sk(X, y, 3, dict(n_estimators=15), cols)
        flat = rc.flatten_forest(rf, cols, 2)
        self.assertEqual(te._parity(rf, flat, X, cols), len(X))
        for i, (f, _, a, b) in enumerate(flat["nodes"]):
            if f != rc.FOREST_LEAF:
                self.assertTrue(a > i and b > i)  # asiklik, sama dg validate() Rust

    def test_selection_uses_training_only_and_no_group_leak(self):
        s, l, g = _data()
        X, yl, ids, _ = te.build_matrix(s, l)
        y = np.array([0 if v == "a" else 1 for v in yl])
        gg = np.array([g[i] for i in ids])
        for tr, te_ in rc.group_folds(y, gg, 0, 3):
            self.assertFalse(set(gg[tr]) & set(gg[te_]))
        a = rc.rank_importance(X[:40], y[:40], gg[:40], "impurity", 0)
        b = rc.rank_importance(X[:40], y[:40], gg[:40], "impurity", 0)
        np.testing.assert_array_equal(a, b)  # deterministik

    def test_trained_requires_group_column(self):
        with tempfile.TemporaryDirectory() as d:
            s, l, g = _data()
            ms.write_csv(d, s, l, {k: k for k in s})
            lab = os.path.join(d, "labels.csv")
            txt = open(lab).read().replace(",group", "").replace("\n", "\n")
            rows = [",".join(r.split(",")[:2]) for r in txt.strip().split("\n")]
            open(lab, "w").write("\n".join(rows) + "\n")
            self.assertFalse(te.labels_have_group_column(lab))
            sessions, labels, groups = te.load_sessions(os.path.join(d, "samples.csv"), lab)
            with self.assertRaises(rc.DataError):
                te.run_rf(sessions, labels, groups, "trained", "v1", os.path.join(d, "m.rs"), None,
                          groups_explicit=False)

    def test_report_separates_synthetic_from_real(self):
        with tempfile.TemporaryDirectory() as d:
            s, l, g = _data()
            rep = te.run_rf(s, l, g, "synthetic_fixture", "v1", os.path.join(d, "m.rs"), None,
                            params=dict(n_estimators=10))
            self.assertIsNone(rep["real_dataset_evaluation"])
            self.assertIsNotNone(rep["pipeline_sanity_check"])
            self.assertEqual(rep["data_provenance"], "synthetic_fixture")
            rep = te.run_rf(s, l, g, "trained", "v1", os.path.join(d, "m.rs"), None,
                            params=dict(n_estimators=10), groups_explicit=True)
            self.assertIsNone(rep["pipeline_sanity_check"])
            self.assertEqual(rep["data_provenance"], "real_dataset")

    def test_feature_selection_inside_training(self):
        with tempfile.TemporaryDirectory() as d:
            s, l, g = _data()
            rep = te.run_rf(s, l, g, "synthetic_fixture", "v1", os.path.join(d, "m.rs"), None,
                            params=dict(n_estimators=10), select_top_k=8)
            self.assertEqual(len(rep["selected_features"]), 8)
            self.assertLessEqual(len(rep["model_size"]["features_used"]), 8)

    def test_feature_order_constants(self):
        self.assertEqual(fr.FEATURE_COUNT, 58)
        self.assertEqual(len(fr.FEATURE_NAMES), 58)
        self.assertEqual(fr.FEATURE_NAMES[:7][1], "gas1_mean")


if __name__ == "__main__":
    unittest.main()
