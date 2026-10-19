//! Format artefak model: regresi logistik multikelas (softmax) linear.
//!
//! Dipilih karena (1) cukup untuk fitur ringkasan per-sesi (58 fitur),
//! (2) tanpa runtime besar, tanpa heap, (3) parameter bisa diekspor dari
//! training offline ke `model_params.rs` dan DIGANTI tanpa menyentuh kode
//! akuisisi/inference. Tidak ada bobot yang dikarang di sini: nilai bobot
//! hanya berasal dari `model_params.rs` yang DIHASILKAN oleh
//! `tools/tinyml/train_export.py`.

use super::features::{FEATURE_COUNT, FEATURE_SPEC};

pub const MAX_CLASSES: usize = 16;
pub const MAX_LABEL_LEN: usize = 12;
pub const MAX_VERSION_LEN: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Dilatih dari dataset berlabel nyata (diekspor dengan --kind trained).
    Trained,
    /// Dilatih dari data SINTETIK untuk menguji pipeline. BUKAN model aroma.
    SyntheticFixture,
}

impl ModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelKind::Trained => "trained",
            ModelKind::SyntheticFixture => "synthetic_fixture",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelError {
    SpecMismatch,
    BadClassCount,
    BadDimensions,
    NonFiniteParam,
    BadStd,
    BadLabel,
    BadVersion,
}

/// Parameter disimpan sebagai slice flat `&'static` (masuk flash, bukan RAM).
pub struct LinearSoftmaxModel {
    pub kind: ModelKind,
    pub version: &'static str,
    /// Harus sama dengan `features::FEATURE_SPEC`.
    pub feature_spec: &'static str,
    pub labels: &'static [&'static str],
    /// Row-major [kelas][fitur], panjang = K * FEATURE_COUNT.
    pub weights: &'static [f32],
    /// Panjang K.
    pub bias: &'static [f32],
    /// Mean z-score dari DATA TRAINING, panjang FEATURE_COUNT.
    pub mean: &'static [f32],
    /// Std z-score dari DATA TRAINING (> 0), panjang FEATURE_COUNT.
    pub std: &'static [f32],
}

fn label_ok(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_LABEL_LEN
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl LinearSoftmaxModel {
    pub fn n_classes(&self) -> usize {
        self.labels.len()
    }

    /// Validasi dimensi/nilai. Dipanggil sebelum setiap prediksi (murah).
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.feature_spec != FEATURE_SPEC {
            return Err(ModelError::SpecMismatch);
        }
        let k = self.labels.len();
        if !(2..=MAX_CLASSES).contains(&k) {
            return Err(ModelError::BadClassCount);
        }
        if self.weights.len() != k * FEATURE_COUNT
            || self.bias.len() != k
            || self.mean.len() != FEATURE_COUNT
            || self.std.len() != FEATURE_COUNT
        {
            return Err(ModelError::BadDimensions);
        }
        if !(self.weights.iter().all(|v| v.is_finite())
            && self.bias.iter().all(|v| v.is_finite())
            && self.mean.iter().all(|v| v.is_finite()))
        {
            return Err(ModelError::NonFiniteParam);
        }
        if !self.std.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(ModelError::BadStd);
        }
        if !self.labels.iter().all(|l| label_ok(l)) {
            return Err(ModelError::BadLabel);
        }
        if !label_ok(self.version) || self.version.len() > MAX_VERSION_LEN {
            return Err(ModelError::BadVersion);
        }
        Ok(())
    }
}
