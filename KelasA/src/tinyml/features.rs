//! Ekstraksi fitur streaming per sesi pengukuran.
//!
//! Satu `FeatureFrame` per detik (8 gas + suhu/RH + status valid) disuap ke
//! `FeatureAccumulator::push()`. RAM konstan (~1 KB), tanpa heap, tanpa
//! menyimpan 300 sample. Setelah sesi selesai, `finalize()` menghasilkan 58 fitur.
//!
//! URUTAN KANAL = urutan payload MQTT:
//!   gas1=MQ-6, gas2=MQ-135, gas3=MQ-7, gas4=MQ-3,
//!   gas5=TGS2602, gas6=TGS2620, gas7=TGS2611, gas8=TGS2600
//! Model hasil training HANYA valid untuk urutan ini. Jika urutan kanal di
//! firmware berubah, training ulang + ganti FEATURE_SPEC.
//! Nilai yang dipakai = TEGANGAN pin ADC (V), bukan ppm.
//!
//! URUTAN FITUR (spec "fv1-8gas7-env2", HARUS identik dgn tools/tinyml/features_ref.py):
//!   untuk c = 0..7 (gas1..gas8): [baseline, mean, std, min, max, delta_rel, slope]
//!   lalu [temp_mean, rh_mean]                                  -> 58 fitur
//! Definisi (hanya sample VALID per kanal):
//!   baseline  = rata-rata BASELINE_N sample valid pertama
//!   mean/std  = rata-rata / simpangan baku populasi (ddof=0)
//!   delta_rel = (mean - baseline) / max(|baseline|, BASELINE_EPS_V)
//!   slope     = kemiringan regresi linear y~t_s (V/detik)
//! Sample tidak valid / non-finite TIDAK pernah diisi nilai pengganti;
//! cukup dilewati dan dihitung. Kalau data valid kurang -> Err, bukan fitur palsu.

use super::mathx::sqrt_f64;

pub const N_GAS: usize = 8;
pub const FEATURES_PER_GAS: usize = 7;
pub const N_ENV_FEATURES: usize = 2;
pub const FEATURE_COUNT: usize = N_GAS * FEATURES_PER_GAS + N_ENV_FEATURES; // 58
pub const FEATURE_SPEC: &str = "fv1-8gas7-env2";

/// Jumlah sample valid pertama per kanal untuk baseline.
pub const BASELINE_N: u32 = 10;
/// Minimum frame yang disuap selama sesi.
pub const MIN_SAMPLES: u32 = 60;
/// Minimum sample valid per kanal gas dan untuk suhu/RH.
pub const MIN_VALID_SAMPLES: u32 = 60;
/// Penyebut delta_rel dijepit minimal ini (Volt) supaya tidak meledak.
pub const BASELINE_EPS_V: f64 = 0.05;
/// DHT22 dianggap valid jika pembacaan sukses terakhir <= ini (detik).
pub const DHT_MAX_AGE_S: u32 = 10;

pub type FeatureVector = [f32; FEATURE_COUNT];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureFrame {
    pub t_s: u32,
    /// Urutan gas1..gas8 (urutan payload), satuan Volt.
    pub gas_v: [f32; N_GAS],
    /// true = nilai kanal ini hasil pembacaan segar pada detik ini.
    pub gas_valid: [bool; N_GAS],
    pub temp_c: f32,
    pub rh_pct: f32,
    pub env_valid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureError {
    /// Frame yang disuap < MIN_SAMPLES.
    InsufficientFrames { have: u32, need: u32 },
    /// Kanal gas (0-based, gas1=0) punya sample valid < MIN_VALID_SAMPLES
    /// atau baseline belum terisi.
    InsufficientValidGas { channel: usize },
    /// Suhu/RH valid < MIN_VALID_SAMPLES.
    InsufficientValidEnv,
    /// Hasil fitur tidak finite (seharusnya tidak terjadi).
    NonFinite,
}

#[derive(Clone, Copy)]
struct ChannelStats {
    n: u32,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    mean_t: f64,
    m2_t: f64,
    cov: f64,
    base_sum: f64,
    base_n: u32,
    rejected: u32,
}

impl ChannelStats {
    const fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean_t: 0.0,
            m2_t: 0.0,
            cov: 0.0,
            base_sum: 0.0,
            base_n: 0,
            rejected: 0,
        }
    }

    fn push(&mut self, t: f64, y: f64) {
        self.n += 1;
        let n = self.n as f64;
        let dt = t - self.mean_t;
        self.mean_t += dt / n;
        let dy = y - self.mean;
        self.mean += dy / n;
        self.m2 += dy * (y - self.mean);
        self.m2_t += dt * (t - self.mean_t);
        self.cov += dt * (y - self.mean);
        if y < self.min {
            self.min = y;
        }
        if y > self.max {
            self.max = y;
        }
        if self.base_n < BASELINE_N {
            self.base_sum += y;
            self.base_n += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct EnvStats {
    n: u32,
    temp_mean: f64,
    rh_mean: f64,
}

pub struct FeatureAccumulator {
    ch: [ChannelStats; N_GAS],
    env: EnvStats,
    frames: u32,
}

impl Default for FeatureAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureAccumulator {
    pub const fn new() -> Self {
        Self {
            ch: [ChannelStats::new(); N_GAS],
            env: EnvStats {
                n: 0,
                temp_mean: 0.0,
                rh_mean: 0.0,
            },
            frames: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// Jumlah sample yang ditolak (tidak valid / non-finite) pada kanal gas c.
    pub fn rejected_gas(&self, channel: usize) -> u32 {
        self.ch.get(channel).map(|c| c.rejected).unwrap_or(0)
    }

    pub fn push(&mut self, f: &FeatureFrame) {
        self.frames = self.frames.saturating_add(1);
        let t = f.t_s as f64;

        for c in 0..N_GAS {
            let v = f.gas_v[c];
            if f.gas_valid[c] && v.is_finite() {
                self.ch[c].push(t, v as f64);
            } else {
                self.ch[c].rejected = self.ch[c].rejected.saturating_add(1);
            }
        }

        if f.env_valid && f.temp_c.is_finite() && f.rh_pct.is_finite() {
            self.env.n += 1;
            let n = self.env.n as f64;
            self.env.temp_mean += (f.temp_c as f64 - self.env.temp_mean) / n;
            self.env.rh_mean += (f.rh_pct as f64 - self.env.rh_mean) / n;
        }
    }

    pub fn finalize(&self) -> Result<FeatureVector, FeatureError> {
        if self.frames < MIN_SAMPLES {
            return Err(FeatureError::InsufficientFrames {
                have: self.frames,
                need: MIN_SAMPLES,
            });
        }
        let mut out = [0.0f32; FEATURE_COUNT];
        let mut k = 0usize;

        for (c, s) in self.ch.iter().enumerate() {
            if s.n < MIN_VALID_SAMPLES || s.base_n < BASELINE_N {
                return Err(FeatureError::InsufficientValidGas { channel: c });
            }
            let n = s.n as f64;
            let baseline = s.base_sum / s.base_n as f64;
            let std = sqrt_f64(s.m2 / n);
            let denom = {
                let a = if baseline < 0.0 { -baseline } else { baseline };
                if a > BASELINE_EPS_V { a } else { BASELINE_EPS_V }
            };
            let delta_rel = (s.mean - baseline) / denom;
            let slope = if s.m2_t > 1e-12 { s.cov / s.m2_t } else { 0.0 };

            let vals = [baseline, s.mean, std, s.min, s.max, delta_rel, slope];
            for v in vals {
                out[k] = v as f32;
                k += 1;
            }
        }

        if self.env.n < MIN_VALID_SAMPLES {
            return Err(FeatureError::InsufficientValidEnv);
        }
        out[k] = self.env.temp_mean as f32;
        out[k + 1] = self.env.rh_mean as f32;

        if out.iter().any(|v| !v.is_finite()) {
            return Err(FeatureError::NonFinite);
        }
        Ok(out)
    }
}
