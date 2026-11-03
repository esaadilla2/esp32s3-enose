# TinyML e-nose (Random Forest) -- training offline, eksperimen G10, ekspor ke firmware

Pipeline firmware (per SESI pengukuran, bukan per sample):
`MeasurementSample -> FeatureFrame -> FeatureAccumulator (streaming) -> 58 fitur
-> Random Forest (soft voting) -> kelas + skor -> JSON MQTT terpisah`.

Definisi fitur: `features_ref.py` (referensi float64) == `src/tinyml/features.rs`
(spec `fv1-8gas7-env2`, urutan per kanal `[baseline, mean, std, min, max, delta_rel, slope]` x gas1..gas8
lalu `[temp_mean, rh_mean]`). Indeks fitur di pohon = indeks ini. Mengubah salah satu =
ubah keduanya + naikkan spec + latih ulang (model lama otomatis ditolak `SpecMismatch`).
Urutan kanal = urutan payload (gas1=MQ-6, gas2=MQ-135, gas3=MQ-7, gas4=MQ-3, gas5=TGS2602,
gas6=TGS2620, gas7=TGS2611, gas8=TGS2600): model yang dilatih dengan urutan lain TIDAK valid.

## Status data (baca dulu)
* Model yang ikut di repo = `synthetic_fixture` (data PALSU dari RNG). Hanya membuktikan pipeline jalan.
  Metriknya (mis. akurasi 1.0) BUKAN akurasi klasifikasi kopi.
* Istilah "model G10" hanya boleh dipakai setelah training+evaluasi pada dataset kopi berlabel (`--kind trained`).
* Laporan JSON memisahkan keduanya: `data_provenance`, `pipeline_sanity_check` (sintetis) vs
  `real_dataset_evaluation` (data nyata); salah satunya selalu `null`.

## Kebutuhan alat host
Python 3 + `numpy`, `scikit-learn`, `scipy` (hanya untuk training/eksperimen; firmware tidak memakai
dependency baru, `Cargo.toml`/`Cargo.lock` tidak berubah).

## Skema dataset
* `samples.csv`: `measurement_id,t_s,gas1..gas8,temperature,humidity[,dht_age_s][,gasN_valid]`
  (tegangan V; sel kosong/nan = tidak valid, dilewati, tidak diganti nilai).
* `labels.csv`: `measurement_id,label,group` -- `group` (alias kolom `Coffee_ID`) = satu biji/lot kopi fisik.
  SEMUA ulangan satu Coffee_ID harus punya group sama. Untuk `--kind trained` kolom ini WAJIB
  (tanpa itu skrip berhenti, supaya tidak bocor antara train/test). Label `[A-Za-z0-9_-]`, maks 12.

## 1. Training + ekspor (Random Forest)
```
python train_export.py --samples samples.csv --labels labels.csv --kind trained --version v1 \
    --out-rs ../../src/tinyml/model_params.rs --report training_report.json \
    --golden-out ../../host-tests/tests/fixtures/golden_data.rs [--n-estimators 50 --max-depth 8 --min-samples-leaf 2]
```
* Split train/test per group (`group_split`), tanpa fitur bocor; fitur mentah, tanpa z-score.
* `--select-top-k K [--select-method impurity|permutation]` (default: tanpa seleksi): peringkat fitur
  dihitung HANYA dari data training (dan diulang di dalam tiap fold CV). Jangan dipakai sebelum
  eksperimen G10 (langkah 2) menunjukkan fitur mana yang stabil.
* Ekspor menulis struktur pohon flat ke `model_params.rs`; threshold dibulatkan KE BAWAH ke float32
  supaya `x <= thr` identik dengan scikit-learn. Skrip memeriksa struktur yang DIEKSPOR vs sklearn
  (gagal = tidak ada file yang dianggap valid) lalu menulis golden fixture (sesi + 192 vektor paritas).
* Laporan memuat hold-out per group + `StratifiedGroupKFold` (group = Coffee_ID) dan ukuran model.

## 2. Eksperimen G10 (>= 20 seed)
```
python g10_experiment.py --samples samples.csv --labels labels.csv --kind trained \
    --seeds 20 --top-k 5,10,15,20,30,58 --out g10_report.json
```
* Tiap seed: `StratifiedGroupKFold(shuffle, seed)` menurut group. Di SETIAP fold, peringkat fitur
  hanya dari training fold: impurity importance, dan permutation importance pada split-dalam
  grup training (test fold tidak disentuh). `--seeds < 20` ditolak.
* Keluaran: Spearman antar peringkat (antar fold/seed), frekuensi fitur masuk top-k, Spearman
  impurity-vs-permutation, importance per kanal gas (informasi saja), dan kinerja saat fitur dikurangi
  ke top-k (dilatih ulang, dinilai pada test fold; mean/std antar seed).
* Peringkat konstan (semua skor sama, mis. permutation importance = 0 di data yang terlalu mudah)
  tidak punya korelasi: dihitung terpisah (`n_constant_rankings`), tidak dipalsukan.
* Skrip hanya MENGUKUR; tidak menghapus fitur/kanal. Keputusan reduksi dibuat setelah hasil nyata dibaca.
* `synthetic_g10_report.json` = lari pada data sintetis (20 seed, 30 pohon): hanya cek pipeline.

## 3. Pengujian
```
python -W ignore -m unittest test_rf_pipeline -v          # Python (7 tes)
cd ../../host-tests && cargo +stable test --target <HOST_TRIPLE>   # Rust di PC
```
Rust: ekstraksi fitur, validasi struktur hutan (siklus/indeks/probabilitas ditolak), hasil hutan
buatan-tangan, status ML, telemetri (<= batas paket), golden Python<->Rust (fitur dari frame + 192
vektor paritas prediksi), RAM/flash artefak. `tests/rf_resources.rs` hanya relevan untuk model RF.

## 4. Sumber daya ESP32-S3 (BELUM diukur)
Tidak ada board/toolchain esp di lingkungan pembuatan paket ini. Yang diketahui secara statis
(`model_size` di report): ukuran artefak model (flash) dan kunjungan node terburuk per inferensi.
Yang HARUS diukur di perangkat: flash firmware total (`cargo +esp size --release`), RAM
(stack/heap saat `classify_session`), dan waktu inferensi, mis. sementara di `acquisition_task`:
```rust
let t0 = embassy_time::Instant::now();
let outcome = classify_session(/* ... */);
println!("ML time = {} us", t0.elapsed().as_micros());
```
Inferensi RF tidak memakai heap; RAM tambahan = `FeatureAccumulator` (672 B) + `FeatureVector` (232 B) + stack kecil.

## Jalur linear lama
`--model linear` pada `train_export.py`/`make_synthetic_fixture.py` masih menghasilkan model softmax
linear (tipe `LinearSoftmaxModel`, trait `Classifier` sama). Setelah mengekspor linear, jalankan
`cargo test --test tinyml_host` (lewati `rf_resources.rs`).

## Fixture sintetis
`make_synthetic_fixture.py --out-rs ... --golden-out ... [--csv-dir DIR]` (data PALSU: 3 kelas x 8 "biji"
x 3 ulangan, group = biji). `model_params.rs` yang tersedia dibuat dengan perintah ini.
