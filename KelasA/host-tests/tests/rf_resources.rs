//! Hanya untuk model_params.rs bertipe RandomForestModel (default).
//! Jalur linear lama: file ini dilewati (jalankan `cargo test --test tinyml_host`).
use tinyml_host_tests::tinyml::inference::Classifier;
use tinyml_host_tests::tinyml::model::FOREST_LEAF;
use tinyml_host_tests::tinyml::model_params::MODEL;

#[test]
fn forest_model_size_and_worst_case_work() {
    // Hitung kedalaman terburuk tiap pohon (hanya di host; tidak ada heap di firmware).
    let n = MODEL.nodes.len();
    let mut depth = vec![0usize; n];
    for i in (0..n).rev() {
        let nd = MODEL.nodes[i];
        depth[i] = if nd.feature == FOREST_LEAF { 0 } else { 1 + depth[nd.left as usize].max(depth[nd.right as usize]) };
    }
    let visits: usize = MODEL.roots.iter().map(|r| depth[*r as usize] + 1).sum();
    println!("RF: pohon={} node={} flash={} B, kunjungan node terburuk/inferensi={}",
        MODEL.roots.len(), n, MODEL.flash_bytes(), visits);
    assert!(MODEL.flash_bytes() < 64 * 1024, "model > 64 KiB");
}

