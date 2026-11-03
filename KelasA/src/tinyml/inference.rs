//! Normalisasi + inference + keputusan status ML. Tanpa heap, tanpa panic
//! (semua indexing lewat iterator/`get`, dimensi divalidasi dulu).

use super::features::{FeatureError, FeatureVector, FEATURE_COUNT};
use super::mathx::exp_nonpositive_f32;
use super::model::{
    LinearSoftmaxModel, ModelError, ModelKind, RandomForestModel, FOREST_LEAF, MAX_CLASSES,
};

/// Antarmuka umum model. main.rs/telemetry hanya bergantung pada trait ini,
/// jadi mengganti keluarga model (RF <-> linear) tidak menyentuh akuisisi.
pub trait Classifier {
    fn kind(&self) -> ModelKind;
    fn version(&self) -> &'static str;
    fn validate(&self) -> Result<(), ModelError>;
    fn predict(&self, features: &FeatureVector) -> Result<Prediction, PredictError>;
    /// Ukuran parameter model di flash (byte).
    fn flash_bytes(&self) -> usize;
}

impl Classifier for LinearSoftmaxModel {
    fn kind(&self) -> ModelKind {
        self.kind
    }
    fn version(&self) -> &'static str {
        self.version
    }
    fn validate(&self) -> Result<(), ModelError> {
        LinearSoftmaxModel::validate(self)
    }
    fn predict(&self, features: &FeatureVector) -> Result<Prediction, PredictError> {
        predict(features, self)
    }
    fn flash_bytes(&self) -> usize {
        (self.weights.len() + self.bias.len() + self.mean.len() + self.std.len()) * 4
    }
}

impl Classifier for RandomForestModel {
    fn kind(&self) -> ModelKind {
        self.kind
    }
    fn version(&self) -> &'static str {
        self.version
    }
    fn validate(&self) -> Result<(), ModelError> {
        RandomForestModel::validate(self)
    }
    fn predict(&self, features: &FeatureVector) -> Result<Prediction, PredictError> {
        predict_forest(features, self)
    }
    fn flash_bytes(&self) -> usize {
        RandomForestModel::flash_bytes(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub class_index: usize,
    pub label: &'static str,
    /// Skor kelas terpilih, 0..1 (linear: softmax; RF: rata-rata suara pohon). BUKAN probabilitas
    /// terkalibrasi; hanya ukuran relatif keyakinan model linear ini.
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictError {
    Model(ModelError),
    NonFiniteInput,
}

/// z-score memakai parameter dari model (hasil training). Tidak dihitung ulang.
pub fn normalize(
    features: &FeatureVector,
    model: &LinearSoftmaxModel,
    out: &mut [f32; FEATURE_COUNT],
) -> Result<(), PredictError> {
    model.validate().map_err(PredictError::Model)?;
    for (((o, x), m), s) in out
        .iter_mut()
        .zip(features.iter())
        .zip(model.mean.iter())
        .zip(model.std.iter())
    {
        if !x.is_finite() {
            return Err(PredictError::NonFiniteInput);
        }
        *o = (*x - *m) / *s;
    }
    Ok(())
}

/// Random Forest: rata-rata distribusi kelas dari daun tiap pohon (soft voting,
/// identik dengan `predict_proba` scikit-learn). Tanpa heap; waktu = jumlah
/// pohon x kedalaman; validasi struktur dilakukan dulu (traversal pasti berhenti).
pub fn predict_forest(
    features: &FeatureVector,
    model: &RandomForestModel,
) -> Result<Prediction, PredictError> {
    model.validate().map_err(PredictError::Model)?;
    if features.iter().any(|v| !v.is_finite()) {
        return Err(PredictError::NonFiniteInput);
    }
    let k = model.n_classes();
    let bad = PredictError::Model(ModelError::BadStructure);
    let mut acc = [0.0f32; MAX_CLASSES];

    for &root in model.roots {
        let mut i = root as usize;
        let mut steps = 0usize;
        loop {
            let node = model.nodes.get(i).ok_or(bad)?;
            if node.feature == FOREST_LEAF {
                let base = (node.left as usize) * k;
                let row = model.leaf_probs.get(base..base + k).ok_or(bad)?;
                for (a, p) in acc.iter_mut().zip(row.iter()) {
                    *a += *p;
                }
                break;
            }
            let x = *features.get(node.feature as usize).ok_or(bad)?;
            i = if x <= node.threshold { node.left } else { node.right } as usize;
            steps += 1;
            if steps > model.nodes.len() {
                return Err(bad);
            }
        }
    }

    let inv = 1.0 / model.roots.len() as f32;
    let mut best = 0usize;
    for c in 1..k {
        if acc[c] > acc[best] {
            best = c;
        }
    }
    Ok(Prediction {
        class_index: best,
        label: model.labels.get(best).copied().ok_or(bad)?,
        score: acc[best] * inv,
    })
}

pub fn predict(
    features: &FeatureVector,
    model: &LinearSoftmaxModel,
) -> Result<Prediction, PredictError> {
    let mut z = [0.0f32; FEATURE_COUNT];
    normalize(features, model, &mut z)?;

    let k = model.n_classes();
    let mut logits = [0.0f32; MAX_CLASSES];
    for (c, row) in model.weights.chunks_exact(FEATURE_COUNT).enumerate() {
        let mut acc = model.bias[c];
        for (w, x) in row.iter().zip(z.iter()) {
            acc += *w * *x;
        }
        if !acc.is_finite() {
            return Err(PredictError::NonFiniteInput);
        }
        logits[c] = acc;
    }

    let mut best = 0usize;
    for c in 1..k {
        if logits[c] > logits[best] {
            best = c;
        }
    }
    let max = logits[best];
    let mut sum = 0.0f32;
    for c in 0..k {
        sum += exp_nonpositive_f32(logits[c] - max);
    }
    let score = 1.0 / sum; // exp(0) = 1 untuk kelas terbaik

    Ok(Prediction {
        class_index: best,
        label: model.labels[best],
        score,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlStatus {
    /// Prediksi dari model Trained pada data sensor fisik.
    Ok,
    /// Prediksi ada, tapi data dummy dan/atau model sintetis: hanya uji integrasi.
    IntegrationTestOnly,
    /// Data valid belum memenuhi syarat minimum -> TIDAK ada prediksi.
    InsufficientData,
    /// Fitur/input non-finite -> TIDAK ada prediksi.
    InvalidFeatures,
    /// Artefak model tidak valid -> TIDAK ada prediksi.
    ModelInvalid,
}

impl MlStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MlStatus::Ok => "ok",
            MlStatus::IntegrationTestOnly => "test_only",
            MlStatus::InsufficientData => "insufficient_data",
            MlStatus::InvalidFeatures => "invalid_features",
            MlStatus::ModelInvalid => "model_invalid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlOutcome {
    pub status: MlStatus,
    pub prediction: Option<Prediction>,
}

/// Gabungkan hasil ekstraksi fitur + model menjadi satu keputusan.
/// Prediksi HANYA ada jika fitur lengkap dan model valid.
pub fn classify_session<M: Classifier>(
    features: Result<FeatureVector, FeatureError>,
    model: &M,
    data_is_dummy: bool,
) -> MlOutcome {
    let fv = match features {
        Ok(f) => f,
        Err(FeatureError::NonFinite) => {
            return MlOutcome {
                status: MlStatus::InvalidFeatures,
                prediction: None,
            }
        }
        Err(_) => {
            return MlOutcome {
                status: MlStatus::InsufficientData,
                prediction: None,
            }
        }
    };
    match model.predict(&fv) {
        Ok(p) => MlOutcome {
            status: if data_is_dummy || model.kind() == ModelKind::SyntheticFixture {
                MlStatus::IntegrationTestOnly
            } else {
                MlStatus::Ok
            },
            prediction: Some(p),
        },
        Err(PredictError::Model(_)) => MlOutcome {
            status: MlStatus::ModelInvalid,
            prediction: None,
        },
        Err(PredictError::NonFiniteInput) => MlOutcome {
            status: MlStatus::InvalidFeatures,
            prediction: None,
        },
    }
}
