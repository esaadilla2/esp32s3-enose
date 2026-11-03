extern crate std;
use std::boxed::Box;
use std::string::String;
use std::vec::Vec;

use tinyml_host_tests::tinyml::features::*;
use tinyml_host_tests::tinyml::inference::*;
use tinyml_host_tests::tinyml::mathx::*;
use tinyml_host_tests::tinyml::model::*;
use tinyml_host_tests::tinyml::model_params::MODEL;
use tinyml_host_tests::tinyml::telemetry::write_ml_json;

#[path = "fixtures/golden_data.rs"]
mod golden_data;

// ---------------------------------------------------------------- helpers
fn frame(t: u32, gas: [f32; N_GAS]) -> FeatureFrame {
    FeatureFrame {
        t_s: t,
        gas_v: gas,
        gas_valid: [true; N_GAS],
        temp_c: 25.0,
        rh_pct: 60.0,
        env_valid: true,
    }
}

fn leak_f32(v: Vec<f32>) -> &'static [f32] {
    Box::leak(v.into_boxed_slice())
}

/// Model UJI (bukan model aroma): 2 kelas, bobot dibuat deterministik.
fn test_model(kind: ModelKind, w0: f32) -> LinearSoftmaxModel {
    let mut w = std::vec![0.0f32; 2 * FEATURE_COUNT];
    w[1] = w0; // kelas 0 bergantung pada fitur #1 (gas1 mean)
    w[FEATURE_COUNT + 1] = -w0; // kelas 1 kebalikannya
    LinearSoftmaxModel {
        kind,
        version: "test-1",
        feature_spec: FEATURE_SPEC,
        labels: Box::leak(Box::new(["pos", "neg"])),
        weights: leak_f32(w),
        bias: leak_f32(std::vec![0.0, 0.0]),
        mean: leak_f32(std::vec![1.0; FEATURE_COUNT]),
        std: leak_f32(std::vec![2.0; FEATURE_COUNT]),
    }
}

// ---------------------------------------------------------------- Random Forest helpers
/// Hutan UJI buatan tangan (bukan model kopi): 2 kelas, 2 pohon.
///  T0: fitur#1 <= 5.0 ? daun0 [1,0] : daun1 [0,1]     T1: satu daun [0.25,0.75]
/// `shift` menggeser threshold (untuk tes fuzz).
fn rf_hand(shift: f32) -> RandomForestModel {
    let nodes: &'static [ForestNode] = Box::leak(Box::new([
        nd(1, 5.0 + shift, 1, 2), // 0: akar T0
        lf(0),                    // 1
        lf(1),                    // 2
        lf(2),                    // 3: akar T1 (daun tunggal)
    ]));
    RandomForestModel {
        kind: ModelKind::Trained,
        version: "rf-test",
        feature_spec: FEATURE_SPEC,
        labels: Box::leak(Box::new(["pos", "neg"])),
        roots: Box::leak(Box::new([0u16, 3])),
        nodes,
        leaf_probs: leak_f32(std::vec![1.0, 0.0, 0.0, 1.0, 0.25, 0.75]),
    }
}

fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= abs + rel * b.abs()
}

// ---------------------------------------------------------------- mathx
#[test]
fn sqrt_matches_std() {
    for &x in &[1e-300, 1e-12, 0.0001, 0.25, 1.0, 2.0, 3.0, 99.9, 1e6, 1e300] {
        assert!(close(sqrt_f64(x), (x as f64).sqrt(), 1e-14, 0.0), "x={x}");
    }
    assert_eq!(sqrt_f64(0.0), 0.0);
    assert!(sqrt_f64(-1.0).is_nan());
    assert!(sqrt_f64(f64::NAN).is_nan());
    assert_eq!(sqrt_f64(f64::INFINITY), f64::INFINITY);
}

#[test]
fn exp_matches_std() {
    let n = 20000;
    for i in 0..=n {
        let x = -87.0f32 * (i as f32) / (n as f32);
        let got = exp_nonpositive_f32(x) as f64;
        let want = (x as f64).exp();
        assert!(close(got, want, 2e-6, 1e-37), "x={x} got={got} want={want}");
    }
    assert_eq!(exp_nonpositive_f32(-200.0), 0.0);
    assert_eq!(exp_nonpositive_f32(5.0), 1.0); // dijepit
    assert!(exp_nonpositive_f32(f32::NAN).is_nan());
}

// ---------------------------------------------------------------- features
#[test]
fn feature_dimension_and_channel_order() {
    assert_eq!(FEATURE_COUNT, 58);
    let mut acc = FeatureAccumulator::new();
    for t in 0..MIN_SAMPLES {
        let mut g = [0.0f32; N_GAS];
        for c in 0..N_GAS {
            g[c] = (c as f32 + 1.0) * 0.5; // gas1=0.5, gas2=1.0, ... gas8=4.0
        }
        acc.push(&frame(t, g));
    }
    let f = acc.finalize().unwrap();
    for c in 0..N_GAS {
        let want = (c as f32 + 1.0) * 0.5;
        let base = c * FEATURES_PER_GAS;
        assert_eq!(f[base], want, "baseline gas{}", c + 1);
        assert_eq!(f[base + 1], want, "mean gas{}", c + 1);
        assert_eq!(f[base + 2], 0.0, "std");
        assert_eq!(f[base + 3], want, "min");
        assert_eq!(f[base + 4], want, "max");
        assert_eq!(f[base + 5], 0.0, "delta_rel");
        assert_eq!(f[base + 6], 0.0, "slope");
    }
    assert_eq!(f[56], 25.0);
    assert_eq!(f[57], 60.0);
}

#[test]
fn windowing_ramp_matches_closed_form() {
    let mut acc = FeatureAccumulator::new();
    for t in 0..100u32 {
        let y = 1.0 + 0.01 * t as f32;
        acc.push(&frame(t, [y; N_GAS]));
    }
    let f = acc.finalize().unwrap();
    let b = &f[0..7];
    assert!(close(b[0] as f64, 1.045, 1e-5, 0.0), "baseline {}", b[0]);
    assert!(close(b[1] as f64, 1.495, 1e-5, 0.0), "mean {}", b[1]);
    assert!(close(b[2] as f64, 0.01 * (((100.0f64 * 100.0) - 1.0) / 12.0).sqrt(), 1e-4, 0.0), "std {}", b[2]);
    assert!(close(b[3] as f64, 1.0, 1e-6, 0.0));
    assert!(close(b[4] as f64, 1.99, 1e-6, 0.0));
    assert!(close(b[5] as f64, (1.495 - 1.045) / 1.045, 1e-4, 0.0), "delta_rel {}", b[5]);
    assert!(close(b[6] as f64, 0.01, 1e-4, 0.0), "slope {}", b[6]);
}

#[test]
fn invalid_samples_are_skipped_not_replaced() {
    let mut acc = FeatureAccumulator::new();
    for t in 0..100u32 {
        let y = 1.0 + 0.01 * t as f32;
        let mut f = frame(t, [y; N_GAS]);
        if t < 20 {
            f.gas_valid[0] = false; // gas1 gagal 20 detik pertama
        }
        if t == 50 {
            f.gas_v[1] = f32::NAN; // NaN juga ditolak walau ditandai valid
        }
        acc.push(&f);
    }
    assert_eq!(acc.rejected_gas(0), 20);
    assert_eq!(acc.rejected_gas(1), 1);
    let f = acc.finalize().unwrap();
    // baseline gas1 = rata-rata t=20..29 -> 1 + 0.01*24.5
    assert!(close(f[0] as f64, 1.245, 1e-5, 0.0), "baseline {}", f[0]);
    assert!(f.iter().all(|v| v.is_finite()));
}

#[test]
fn insufficient_data_is_error_not_fake_features() {
    // terlalu sedikit frame
    let mut acc = FeatureAccumulator::new();
    for t in 0..(MIN_SAMPLES - 1) {
        acc.push(&frame(t, [1.0; N_GAS]));
    }
    assert_eq!(
        acc.finalize(),
        Err(FeatureError::InsufficientFrames { have: MIN_SAMPLES - 1, need: MIN_SAMPLES })
    );
    // satu kanal mati total
    let mut acc = FeatureAccumulator::new();
    for t in 0..120u32 {
        let mut f = frame(t, [1.0; N_GAS]);
        f.gas_valid[4] = false;
        acc.push(&f);
    }
    assert_eq!(acc.finalize(), Err(FeatureError::InsufficientValidGas { channel: 4 }));
    // suhu/RH tidak valid
    let mut acc = FeatureAccumulator::new();
    for t in 0..120u32 {
        let mut f = frame(t, [1.0; N_GAS]);
        f.env_valid = false;
        acc.push(&f);
    }
    assert_eq!(acc.finalize(), Err(FeatureError::InsufficientValidEnv));
    // reset
    acc.reset();
    assert_eq!(acc.frames(), 0);
}

// ---------------------------------------------------------------- normalisasi + model
#[test]
fn normalization_uses_model_params() {
    let m = test_model(ModelKind::Trained, 1.0);
    let x: FeatureVector = [5.0; FEATURE_COUNT];
    let mut z = [0.0f32; FEATURE_COUNT];
    normalize(&x, &m, &mut z).unwrap();
    assert!(z.iter().all(|v| *v == 2.0)); // (5-1)/2
}

#[test]
fn predict_is_deterministic_and_score_is_probability() {
    let m = test_model(ModelKind::Trained, 1.0);
    let mut x: FeatureVector = [1.0; FEATURE_COUNT];
    x[1] = 9.0; // z = 4 -> logit kelas0=4, kelas1=-4
    let a = predict(&x, &m).unwrap();
    let b = predict(&x, &m).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.label, "pos");
    let want = 1.0 / (1.0 + (-8.0f64).exp());
    assert!(close(a.score as f64, want, 1e-5, 0.0), "score {}", a.score);
    x[1] = -7.0; // z = -4 -> kelas1
    let c = predict(&x, &m).unwrap();
    assert_eq!(c.label, "neg");
}

#[test]
fn invalid_input_and_model_are_rejected() {
    let m = test_model(ModelKind::Trained, 1.0);
    let mut x: FeatureVector = [1.0; FEATURE_COUNT];
    x[3] = f32::NAN;
    assert_eq!(predict(&x, &m), Err(PredictError::NonFiniteInput));
    x[3] = f32::INFINITY;
    assert_eq!(predict(&x, &m), Err(PredictError::NonFiniteInput));

    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.feature_spec = "other";
    assert_eq!(bad.validate(), Err(ModelError::SpecMismatch));
    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.std = leak_f32(std::vec![0.0; FEATURE_COUNT]);
    assert_eq!(bad.validate(), Err(ModelError::BadStd));
    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.labels = Box::leak(Box::new(["only_one"]));
    assert_eq!(bad.validate(), Err(ModelError::BadClassCount));
    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.labels = Box::leak(Box::new(["ok", "ba\"d"]));
    assert_eq!(bad.validate(), Err(ModelError::BadLabel));
    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.weights = leak_f32(std::vec![0.0; 3]);
    assert_eq!(bad.validate(), Err(ModelError::BadDimensions));
    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.bias = leak_f32(std::vec![f32::NAN, 0.0]);
    assert_eq!(bad.validate(), Err(ModelError::NonFiniteParam));
}

#[test]
fn predict_never_panics_on_extreme_inputs() {
    let m = test_model(ModelKind::Trained, 1.0);
    let mut s: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    for _ in 0..20000 {
        let mut x: FeatureVector = [0.0; FEATURE_COUNT];
        for v in x.iter_mut() {
            *v = match next() % 6 {
                0 => f32::from_bits(next()), // bit acak: bisa NaN/inf/subnormal
                1 => f32::MAX,
                2 => -f32::MAX,
                3 => 1e-38,
                _ => (next() as f32 / u32::MAX as f32) * 100.0 - 50.0,
            };
        }
        let _ = predict(&x, &m); // tidak boleh panic
        let _ = MODEL.predict(&x);
        let _ = predict_forest(&x, &rf_hand(0.0));
    }
}

#[test]
fn classify_session_status_rules() {
    let trained = test_model(ModelKind::Trained, 1.0);
    let synth = test_model(ModelKind::SyntheticFixture, 1.0);
    let fv: FeatureVector = [1.0; FEATURE_COUNT];

    let o = classify_session(Ok(fv), &trained, false);
    assert_eq!(o.status, MlStatus::Ok);
    assert!(o.prediction.is_some());

    let o = classify_session(Ok(fv), &trained, true); // data dummy
    assert_eq!(o.status, MlStatus::IntegrationTestOnly);
    assert!(o.prediction.is_some());

    let o = classify_session(Ok(fv), &synth, false); // model sintetis
    assert_eq!(o.status, MlStatus::IntegrationTestOnly);

    let o = classify_session(Err(FeatureError::InsufficientValidEnv), &trained, false);
    assert_eq!((o.status, o.prediction), (MlStatus::InsufficientData, None));
    let o = classify_session(Err(FeatureError::InsufficientFrames { have: 3, need: 60 }), &trained, false);
    assert_eq!((o.status, o.prediction), (MlStatus::InsufficientData, None));
    let o = classify_session(Err(FeatureError::NonFinite), &trained, false);
    assert_eq!((o.status, o.prediction), (MlStatus::InvalidFeatures, None));

    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.feature_spec = "x";
    let o = classify_session(Ok(fv), &bad, false);
    assert_eq!((o.status, o.prediction), (MlStatus::ModelInvalid, None));
}

// ---------------------------------------------------------------- Random Forest
#[test]
fn forest_hand_built_matches_manual_computation() {
    let m = rf_hand(0.0);
    assert!(m.validate().is_ok());
    let mut x: FeatureVector = [0.0; FEATURE_COUNT];
    x[1] = 3.0; // T0 -> [1,0]; T1 -> [.25,.75]  => [.625,.375]
    let p = m.predict(&x).unwrap();
    assert_eq!((p.class_index, p.label), (0, "pos"));
    assert!(close(p.score as f64, 0.625, 1e-7, 0.0), "{}", p.score);
    x[1] = 5.0; // batas: x <= thr -> kiri (sama dengan sklearn)
    assert_eq!(m.predict(&x).unwrap().class_index, 0);
    x[1] = 5.000001;
    let p = m.predict(&x).unwrap(); // T0 -> [0,1] => [.125,.875]
    assert_eq!((p.class_index, p.label), (1, "neg"));
    assert!(close(p.score as f64, 0.875, 1e-6, 0.0), "{}", p.score);
}

#[test]
fn forest_invalid_structure_is_rejected_never_followed() {
    let base = rf_hand(0.0);
    let mk = |nodes: Vec<ForestNode>, roots: Vec<u16>, probs: Vec<f32>| RandomForestModel {
        nodes: Box::leak(nodes.into_boxed_slice()),
        roots: Box::leak(roots.into_boxed_slice()),
        leaf_probs: leak_f32(probs),
        ..rf_hand(0.0)
    };
    let probs = || std::vec![1.0, 0.0, 0.0, 1.0, 0.25, 0.75];
    // siklus: anak menunjuk ke induk / dirinya sendiri
    assert_eq!(mk(std::vec![nd(1, 5.0, 0, 2), lf(0), lf(1), lf(2)], std::vec![0, 3], probs()).validate(), Err(ModelError::BadStructure));
    // anak di luar array
    assert_eq!(mk(std::vec![nd(1, 5.0, 1, 9), lf(0), lf(1), lf(2)], std::vec![0, 3], probs()).validate(), Err(ModelError::BadStructure));
    // indeks daun di luar leaf_probs
    assert_eq!(mk(std::vec![nd(1, 5.0, 1, 2), lf(0), lf(7), lf(2)], std::vec![0, 3], probs()).validate(), Err(ModelError::BadStructure));
    // fitur di luar 58
    assert_eq!(mk(std::vec![nd(58, 5.0, 1, 2), lf(0), lf(1), lf(2)], std::vec![0, 3], probs()).validate(), Err(ModelError::BadStructure));
    // akar di luar array
    assert_eq!(mk(std::vec![nd(1, 5.0, 1, 2), lf(0), lf(1), lf(2)], std::vec![0, 4], probs()).validate(), Err(ModelError::BadStructure));
    // tanpa pohon
    assert_eq!(mk(std::vec![nd(1, 5.0, 1, 2), lf(0), lf(1), lf(2)], std::vec![], probs()).validate(), Err(ModelError::BadStructure));
    // threshold NaN
    assert_eq!(mk(std::vec![nd(1, f32::NAN, 1, 2), lf(0), lf(1), lf(2)], std::vec![0, 3], probs()).validate(), Err(ModelError::NonFiniteParam));
    // probabilitas di luar [0,1] / ukuran tidak kelipatan K
    assert_eq!(mk(std::vec![nd(1, 5.0, 1, 2), lf(0), lf(1), lf(2)], std::vec![0, 3], std::vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.75]).validate(), Err(ModelError::NonFiniteParam));
    assert_eq!(mk(std::vec![nd(1, 5.0, 1, 2), lf(0), lf(1), lf(2)], std::vec![0, 3], std::vec![1.0, 0.0, 0.0]).validate(), Err(ModelError::BadDimensions));
    // spec / label salah
    let mut bad = rf_hand(0.0);
    bad.feature_spec = "other";
    assert_eq!(bad.validate(), Err(ModelError::SpecMismatch));
    let mut bad = rf_hand(0.0);
    bad.labels = Box::leak(Box::new(["ok", "ba\"d"]));
    assert_eq!(bad.validate(), Err(ModelError::BadLabel));
    // predict menolak model rusak & input non-finite
    let x: FeatureVector = [1.0; FEATURE_COUNT];
    let broken = mk(std::vec![nd(1, 5.0, 0, 2), lf(0), lf(1), lf(2)], std::vec![0, 3], probs());
    assert!(matches!(broken.predict(&x), Err(PredictError::Model(ModelError::BadStructure))));
    let mut nan = x;
    nan[0] = f32::NAN;
    assert_eq!(base.predict(&nan), Err(PredictError::NonFiniteInput));
}

#[test]
fn forest_classify_session_status_rules() {
    let rf = rf_hand(0.0);
    let fv: FeatureVector = [1.0; FEATURE_COUNT];
    let o = classify_session(Ok(fv), &rf, false);
    assert_eq!(o.status, MlStatus::Ok);
    assert!(o.prediction.is_some());
    assert_eq!(classify_session(Ok(fv), &rf, true).status, MlStatus::IntegrationTestOnly);
    let mut syn = rf_hand(0.0);
    syn.kind = ModelKind::SyntheticFixture;
    assert_eq!(classify_session(Ok(fv), &syn, false).status, MlStatus::IntegrationTestOnly);
    let o = classify_session(Err(FeatureError::InsufficientValidEnv), &rf, false);
    assert_eq!((o.status, o.prediction), (MlStatus::InsufficientData, None));
    let mut bad = rf_hand(0.0);
    bad.feature_spec = "x";
    let o = classify_session(Ok(fv), &bad, false);
    assert_eq!((o.status, o.prediction), (MlStatus::ModelInvalid, None));
}

#[test]
fn forest_telemetry_json_shape() {
    let rf = rf_hand(0.0);
    let fv: FeatureVector = [1.0; FEATURE_COUNT];
    let o = classify_session(Ok(fv), &rf, false);
    let mut w = W(String::new());
    write_ml_json(&mut w, "X1", "hardware", &rf, &o, 300).unwrap();
    std::println!("JSON_RF={}", w.0);
    assert!(json_shape_ok(&w.0));
    for k in ["\"ml_status\":\"ok\"", "\"model_version\":\"rf-test\"", "\"predicted_class\":\"", "\"prediction_score\":"] {
        assert!(w.0.contains(k), "{k}: {}", w.0);
    }
}

/// Paritas prediksi Python (struktur ekspor, float32, urutan akumulasi sama) <-> Rust,
/// atas vektor fitur langsung (data asli + varian terjitter) -- menjangkau banyak cabang pohon.
#[test]
fn python_rust_prediction_parity_on_feature_vectors() {
    assert!(!golden_data::PARITY.is_empty());
    let mut classes = std::collections::BTreeSet::new();
    for (i, v) in golden_data::PARITY.iter().enumerate() {
        let p = MODEL.predict(&v.x).unwrap();
        assert_eq!(p.class_index, v.class_index, "PARITY#{i} kelas");
        assert!(close(p.score as f64, v.score, 1e-6, 1e-6), "PARITY#{i} skor rust={} py={}", p.score, v.score);
        classes.insert(p.class_index);
    }
    std::println!("PARITY vektor={} kelas_terprediksi={}", golden_data::PARITY.len(), classes.len());
}

// ---------------------------------------------------------------- golden (Python <-> Rust)
#[test]
fn golden_python_reference_matches_firmware() {
    assert_eq!(golden_data::GOLDEN_FEATURE_SPEC, FEATURE_SPEC);
    assert!(Classifier::validate(&MODEL).is_ok());
    assert!(!golden_data::GOLDEN.is_empty());
    let (mut n_ok, mut n_err) = (0, 0);
    for s in golden_data::GOLDEN {
        let mut acc = FeatureAccumulator::new();
        for g in s.frames {
            let mut gv = [false; N_GAS];
            for c in 0..N_GAS {
                gv[c] = (g.valid >> c) & 1 == 1;
            }
            acc.push(&FeatureFrame {
                t_s: g.t_s,
                gas_v: g.gas,
                gas_valid: gv,
                temp_c: g.temp,
                rh_pct: g.rh,
                env_valid: g.env_ok,
            });
        }
        let res = acc.finalize();
        if s.expected_error.is_empty() {
            n_ok += 1;
            let f = res.unwrap_or_else(|e| panic!("{}: tak terduga {:?}", s.name, e));
            assert_eq!(s.features.len(), FEATURE_COUNT);
            for (i, (a, b)) in f.iter().zip(s.features.iter()).enumerate() {
                assert!(close(*a as f64, *b, 1e-4, 1e-5), "{} fitur#{i}: rust={a} py={b}", s.name);
            }
            let p = MODEL.predict(&f).unwrap();
            assert_eq!(p.class_index, s.class_index, "{} kelas", s.name);
            assert!(close(p.score as f64, s.score, 1e-3, 1e-4), "{} skor rust={} py={}", s.name, p.score, s.score);
        } else {
            n_err += 1;
            let want = s.expected_error;
            match res {
                Err(FeatureError::InsufficientFrames { .. }) => assert_eq!(want, "insufficient_frames", "{}", s.name),
                Err(FeatureError::InsufficientValidEnv) => assert_eq!(want, "insufficient_valid_env", "{}", s.name),
                Err(FeatureError::InsufficientValidGas { channel }) => {
                    assert_eq!(want, std::format!("insufficient_valid_gas:{channel}"), "{}", s.name)
                }
                other => panic!("{}: harapan {want}, dapat {:?}", s.name, other),
            }
        }
    }
    assert!(n_ok >= 1, "ok={n_ok} err={n_err}");
    // Fixture sintetis menyertakan kasus tepi (prefix "edge_"); dataset nyata boleh tidak.
    if golden_data::GOLDEN.iter().any(|s| s.name.starts_with("edge_")) {
        assert!(n_ok >= 3 && n_err >= 2, "ok={n_ok} err={n_err}");
    }
}

// ---------------------------------------------------------------- telemetry
struct W(String);
impl core::fmt::Write for W {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.0.push_str(s);
        Ok(())
    }
}

fn json_shape_ok(s: &str) -> bool {
    s.starts_with('{') && s.ends_with('}') && s.matches('"').count() % 2 == 0 && !s.contains(",}") && !s.contains(",,")
}

#[test]
fn telemetry_has_explicit_status_and_no_fake_label() {
    let m = test_model(ModelKind::Trained, 1.0);
    let fv: FeatureVector = [1.0; FEATURE_COUNT];

    // prediksi ada
    let o = classify_session(Ok(fv), &m, true);
    let mut w = W(String::new());
    write_ml_json(&mut w, "DAQ03_B02_C07_A04_M01", "dummy", &m, &o, 300).unwrap();
    std::println!("JSON_OK={}", w.0);
    assert!(json_shape_ok(&w.0));
    for k in ["\"measurement_id\":\"DAQ03_B02_C07_A04_M01\"", "\"ml_status\":\"test_only\"",
              "\"data_mode\":\"dummy\"", "\"ml_frames\":300",
              "\"model_version\":\"test-1\"", "\"predicted_class\":\"", "\"prediction_score\":"] {
        assert!(w.0.contains(k), "hilang {k}: {}", w.0);
    }

    // tanpa prediksi -> TIDAK ada predicted_class / prediction_score
    let o = classify_session(Err(FeatureError::InsufficientValidEnv), &m, false);
    let mut w = W(String::new());
    write_ml_json(&mut w, "X", "hardware", &m, &o, 12).unwrap();
    std::println!("JSON_NOPRED={}", w.0);
    assert!(json_shape_ok(&w.0));
    assert!(w.0.contains("\"ml_status\":\"insufficient_data\""));
    assert!(!w.0.contains("synthetic"));
    assert!(!w.0.contains("predicted_class") && !w.0.contains("prediction_score"));

    // model tidak valid -> versi tidak dicetak
    let mut bad = test_model(ModelKind::Trained, 1.0);
    bad.version = "bad\"ver";
    let o = classify_session(Ok(fv), &bad, false);
    let mut w = W(String::new());
    write_ml_json(&mut w, "X", "hardware", &bad, &o, 300).unwrap();
    assert!(json_shape_ok(&w.0) && !w.0.contains("model_version"));
}

#[test]
fn telemetry_fits_mqtt_packet_limit_worst_case() {
    // Batas embassy-mqtt-lite 0.3: MAX_PACKET_SIZE = 256 (asumsi: total paket
    // termasuk header + topik "v1/devices/me/telemetry" = 23 byte).
    let labels: &'static [&'static str] = Box::leak(Box::new(["aaaaaaaaaaaa", "bbbbbbbbbbbb"]));
    let mut m = test_model(ModelKind::SyntheticFixture, 1.0);
    m.labels = labels;
    m.version = "vvvvvvvvvvvv";
    let fv: FeatureVector = [1.0; FEATURE_COUNT];
    let o = classify_session(Ok(fv), &m, true);
    let mid = "M".repeat(32); // MEASUREMENT_ID_CAPACITY
    let mut w = W(String::new());
    write_ml_json(&mut w, &mid, "hardware", &m, &o, 300).unwrap();
    let packet = 1 + 2 + 2 + 23 + w.0.len();
    std::println!("worst-case ML payload={} bytes, paket~{}", w.0.len(), packet);
    assert!(packet <= 256, "paket {packet} > 256");
}

// ---------------------------------------------------------------- memori
#[test]
fn ram_footprint_is_small_and_fixed() {
    let acc = core::mem::size_of::<FeatureAccumulator>();
    let fv = core::mem::size_of::<FeatureVector>();
    let flash_params = MODEL.flash_bytes();
    std::println!("FeatureAccumulator={acc} B, FeatureVector={fv} B, parameter model(flash)={flash_params} B");
    assert!(acc <= 1024, "akumulator {acc} B");
    assert_eq!(fv, FEATURE_COUNT * 4);
}
