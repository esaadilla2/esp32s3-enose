//! Normalisasi + inference + keputusan status ML. Tanpa heap, tanpa panic
//! (semua indexing lewat iterator/`get`, dimensi divalidasi dulu).

use super::features::{FeatureError, FeatureVector, FEATURE_COUNT};
use super::mathx::exp_nonpositive_f32;
use super::model::{LinearSoftmaxModel, ModelError, ModelKind, MAX_CLASSES};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub class_index: usize,
    pub label: &'static str,
    /// Probabilitas softmax kelas terpilih, 0..1. BUKAN probabilitas
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
pub fn classify_session(
    features: Result<FeatureVector, FeatureError>,
    model: &LinearSoftmaxModel,
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
    match predict(&fv, model) {
        Ok(p) => MlOutcome {
            status: if data_is_dummy || model.kind == ModelKind::SyntheticFixture {
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
