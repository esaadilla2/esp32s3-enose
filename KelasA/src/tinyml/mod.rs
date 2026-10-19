//! TinyML untuk klasifikasi pola aroma kopi (ESP32-S3 e-nose).
//!
//! Pipeline (satu keputusan per SESI pengukuran, bukan per sample):
//!
//! ```text
//! MeasurementSample (acquisition) -> FeatureFrame (8 gas + suhu/RH + status valid)
//!   -> FeatureAccumulator::push()      [streaming, O(1) RAM, tiap detik]
//!   -> FeatureAccumulator::finalize()  [58 fitur, setelah sesi selesai]
//!   -> normalisasi z-score (parameter dari training, model_params.rs)
//!   -> LinearSoftmaxModel -> kelas + skor
//!   -> JSON telemetry terpisah (MQTT)
//! ```
//!
//! Semua modul di sini murni `core` (tanpa heapless/esp-hal/alloc) supaya bisa
//! di-host-test lewat crate `host-tests/` dengan source yang SAMA persis.
//!
//! PENTING: hasil inference pada data dummy atau model `SyntheticFixture`
//! hanya menguji integrasi teknis, BUKAN akurasi klasifikasi aroma.

pub mod features;
pub mod inference;
pub mod mathx;
pub mod model;
pub mod model_params;
pub mod telemetry;
pub mod model_rf;
pub mod inference_rf;