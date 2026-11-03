//! Format artefak model. Dua keluarga model, keduanya `no_std`, tanpa heap:
//!
//! * `RandomForestModel` (DEFAULT, G10): pohon keputusan flat + voting rata-rata
//!   probabilitas daun (sama dengan `predict_proba` scikit-learn).
//! * `LinearSoftmaxModel` (dipertahankan sebagai jalur lama): softmax linear.
//!
//! Keduanya diakses lewat trait `inference::Classifier`, jadi main.rs dan
//! acquisition.rs tidak berubah saat model diganti. Tidak ada parameter yang
//! dikarang di sini: semuanya HANYA berasal dari `model_params.rs` yang
//! DIHASILKAN oleh `tools/tinyml/train_export.py`.

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
    /// Struktur Random Forest rusak (indeks keluar batas, anak <= induk, dst).
    BadStructure,
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

// =====================================================================
// RANDOM FOREST
// =====================================================================

/// Penanda node daun pada `ForestNode::feature`.
pub const FOREST_LEAF: u8 = 0xFF;
pub const MAX_TREES: usize = 256;
/// Indeks node disimpan sebagai u16.
pub const MAX_NODES: usize = 60_000;

/// Satu node pohon (12 byte). Semua pohon disimpan flat dalam satu array.
///  * internal: `feature` < FEATURE_COUNT; fitur <= `threshold` -> `left`, selain itu `right`
///    (`left`/`right` = indeks ABSOLUT di array node, selalu > indeks node ini)
///  * daun: `feature` == FOREST_LEAF; `left` = indeks daun di `leaf_probs` (K nilai per daun)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForestNode {
    pub feature: u8,
    pub threshold: f32,
    pub left: u16,
    pub right: u16,
}

/// Konstruktor ringkas untuk `model_params.rs` (hasil generate).
pub const fn nd(feature: u8, threshold: f32, left: u16, right: u16) -> ForestNode {
    ForestNode { feature, threshold, left, right }
}
pub const fn lf(leaf_index: u16) -> ForestNode {
    ForestNode { feature: FOREST_LEAF, threshold: 0.0, left: leaf_index, right: 0 }
}

/// Random Forest hasil training offline. Fitur MENTAH (tanpa z-score): pohon
/// keputusan tidak berubah terhadap penskalaan monoton, jadi normalisasi
/// tidak diperlukan pada keluarga model ini.
pub struct RandomForestModel {
    pub kind: ModelKind,
    pub version: &'static str,
    /// Harus sama dengan `features::FEATURE_SPEC`.
    pub feature_spec: &'static str,
    pub labels: &'static [&'static str],
    /// Indeks node akar tiap pohon.
    pub roots: &'static [u16],
    pub nodes: &'static [ForestNode],
    /// Row-major [daun][kelas]; tiap baris = distribusi kelas (jumlah 1).
    pub leaf_probs: &'static [f32],
}

impl RandomForestModel {
    pub fn n_classes(&self) -> usize {
        self.labels.len()
    }

    /// Ukuran artefak di flash (byte): node + probabilitas daun + akar.
    pub fn flash_bytes(&self) -> usize {
        self.nodes.len() * core::mem::size_of::<ForestNode>()
            + self.leaf_probs.len() * 4
            + self.roots.len() * 2
    }

    /// Validasi struktur penuh: setelah lolos, traversal TIDAK bisa keluar batas
    /// atau berputar (anak selalu berindeks > induk).
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.feature_spec != FEATURE_SPEC {
            return Err(ModelError::SpecMismatch);
        }
        let k = self.labels.len();
        if !(2..=MAX_CLASSES).contains(&k) {
            return Err(ModelError::BadClassCount);
        }
        if !self.labels.iter().all(|l| label_ok(l)) {
            return Err(ModelError::BadLabel);
        }
        if !label_ok(self.version) || self.version.len() > MAX_VERSION_LEN {
            return Err(ModelError::BadVersion);
        }
        if self.roots.is_empty() || self.roots.len() > MAX_TREES {
            return Err(ModelError::BadStructure);
        }
        let n = self.nodes.len();
        if n == 0 || n > MAX_NODES {
            return Err(ModelError::BadStructure);
        }
        if self.leaf_probs.is_empty() || self.leaf_probs.len() % k != 0 {
            return Err(ModelError::BadDimensions);
        }
        if !self.leaf_probs.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0) {
            return Err(ModelError::NonFiniteParam);
        }
        let n_leaves = self.leaf_probs.len() / k;
        for (i, node) in self.nodes.iter().enumerate() {
            if node.feature == FOREST_LEAF {
                if (node.left as usize) >= n_leaves {
                    return Err(ModelError::BadStructure);
                }
            } else {
                if (node.feature as usize) >= FEATURE_COUNT {
                    return Err(ModelError::BadStructure);
                }
                if !node.threshold.is_finite() {
                    return Err(ModelError::NonFiniteParam);
                }
                let (l, r) = (node.left as usize, node.right as usize);
                if l <= i || r <= i || l >= n || r >= n {
                    return Err(ModelError::BadStructure);
                }
            }
        }
        if !self.roots.iter().all(|r| (*r as usize) < n) {
            return Err(ModelError::BadStructure);
        }
        Ok(())
    }
}
