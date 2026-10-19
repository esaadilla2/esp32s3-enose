"""Referensi fitur (float64) -- HARUS identik dengan src/tinyml/features.rs.

Spec "fv1-8gas7-env2". Urutan kanal = urutan payload MQTT gas1..gas8
(gas1=MQ-6, gas2=MQ-135, gas3=MQ-7, gas4=MQ-3, gas5=TGS2602, gas6=TGS2620,
gas7=TGS2611, gas8=TGS2600). Nilai = tegangan pin ADC (V).
Fitur: per kanal [baseline, mean, std, min, max, delta_rel, slope] x 8,
lalu [temp_mean, rh_mean] -> 58.
"""
import numpy as np

FEATURE_SPEC = "fv1-8gas7-env2"
N_GAS = 8
FEATURES_PER_GAS = 7
FEATURE_COUNT = N_GAS * FEATURES_PER_GAS + 2
BASELINE_N = 10
MIN_SAMPLES = 60
MIN_VALID_SAMPLES = 60
BASELINE_EPS_V = 0.05
DHT_MAX_AGE_S = 10

GAS_NAMES = ["MQ-6", "MQ-135", "MQ-7", "MQ-3", "TGS2602", "TGS2620", "TGS2611", "TGS2600"]
STAT_NAMES = ["baseline", "mean", "std", "min", "max", "delta_rel", "slope"]
FEATURE_NAMES = [f"gas{c+1}_{s}" for c in range(N_GAS) for s in STAT_NAMES] + ["temp_mean", "rh_mean"]
assert len(FEATURE_NAMES) == FEATURE_COUNT


class FeatureError(Exception):
    """kind: insufficient_frames | insufficient_valid_gas:<ch> | insufficient_valid_env"""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


def _f32(a):
    # Firmware menerima f32; samakan presisi input sebelum dihitung dalam f64.
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def session_features(t_s, gas_v, gas_valid, temp_c, rh_pct, env_valid):
    """t_s (n,), gas_v (n,8), gas_valid (n,8) bool, temp_c/rh_pct (n,), env_valid (n,) bool."""
    t_s = np.asarray(t_s, dtype=np.float64)
    gas_v = _f32(gas_v)
    gas_valid = np.asarray(gas_valid, dtype=bool)
    temp_c, rh_pct = _f32(temp_c), _f32(rh_pct)
    env_valid = np.asarray(env_valid, dtype=bool)
    n = len(t_s)
    if n < MIN_SAMPLES:
        raise FeatureError("insufficient_frames")

    out = []
    for c in range(N_GAS):
        ok = gas_valid[:, c] & np.isfinite(gas_v[:, c])
        y, t = gas_v[ok, c], t_s[ok]
        if len(y) < MIN_VALID_SAMPLES or len(y) < BASELINE_N:
            raise FeatureError(f"insufficient_valid_gas:{c}")
        baseline = y[:BASELINE_N].mean()
        mean = y.mean()
        std = y.std()  # ddof=0
        denom = max(abs(baseline), BASELINE_EPS_V)
        tc = t - t.mean()
        m2t = float((tc * tc).sum())
        slope = float((tc * (y - mean)).sum() / m2t) if m2t > 1e-12 else 0.0
        out += [baseline, mean, std, y.min(), y.max(), (mean - baseline) / denom, slope]

    eok = env_valid & np.isfinite(temp_c) & np.isfinite(rh_pct)
    if eok.sum() < MIN_VALID_SAMPLES:
        raise FeatureError("insufficient_valid_env")
    out += [temp_c[eok].mean(), rh_pct[eok].mean()]
    return np.asarray(out, dtype=np.float64)
