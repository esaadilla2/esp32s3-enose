//! JSON telemetry hasil ML. DIKIRIM SEBAGAI PESAN TERPISAH (sekali per sesi),
//! BUKAN ditambahkan ke payload per-sample: payload per-sample sudah ~216-221
//! byte dan batas paket embassy-mqtt-lite = 256 byte, jadi menambah field di
//! sana akan membuat publish gagal. Karena itu key sengaja ringkas dan
//! `model_kind` TIDAK dikirim: model sintetis ditandai `"synthetic":true`,
//! status `test_only` = prediksi hanya uji integrasi (data dummy / model sintetis).

use core::fmt::Write;

use super::inference::MlOutcome;
use super::model::{LinearSoftmaxModel, ModelKind};

/// Tulis JSON ke `w`. `measurement_id`/`data_mode` harus ASCII aman JSON
/// (dibangun firmware dari angka & konstanta). Label/versi model sudah
/// divalidasi hanya [A-Za-z0-9_-] oleh `LinearSoftmaxModel::validate()`;
/// kalau model tidak valid, versi/label TIDAK dicetak.
pub fn write_ml_json<W: Write>(
    w: &mut W,
    measurement_id: &str,
    data_mode: &str,
    model: &LinearSoftmaxModel,
    outcome: &MlOutcome,
    n_frames: u32,
) -> core::fmt::Result {
    let model_ok = model.validate().is_ok();
    write!(
        w,
        "{{\"measurement_id\":\"{}\",\"ml_status\":\"{}\",\"data_mode\":\"{}\",\"ml_frames\":{}",
        measurement_id,
        outcome.status.as_str(),
        data_mode,
        n_frames
    )?;
    if model.kind == ModelKind::SyntheticFixture {
        w.write_str(",\"synthetic\":true")?;
    }
    if model_ok {
        write!(w, ",\"model_version\":\"{}\"", model.version)?;
    }
    if let Some(p) = outcome.prediction {
        write!(
            w,
            ",\"predicted_class\":\"{}\",\"prediction_score\":{:.4}",
            p.label, p.score
        )?;
    }
    w.write_str("}")
}
