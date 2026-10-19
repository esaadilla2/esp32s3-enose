# TinyML e-nose -- training offline & ekspor ke firmware

Pipeline firmware (per SESI pengukuran, bukan per sample):
`MeasurementSample -> FeatureFrame -> FeatureAccumulator (streaming) -> 58 fitur
-> z-score (parameter training) -> softmax linear -> kelas + skor -> JSON MQTT terpisah`.

Definisi fitur: `features_ref.py` (referensi, float64) == `src/tinyml/features.rs`
(spec `fv1-8gas7-env2`). Mengubah salah satu = ubah keduanya + naikkan spec + regenerasi.

## Skema dataset (lihat docstring `train_export.py`)
- `samples.csv`: `measurement_id,t_s,gas1..gas8,temperature,humidity[,dht_age_s][,gasN_valid]`
  gas1..gas8 = tegangan (V) urutan payload MQTT (MQ-6, MQ-135, MQ-7, MQ-3, TGS2602, TGS2620, TGS2611, TGS2600).
  Sel kosong/nan = tidak valid (dilewati, tidak diganti nilai).
- `labels.csv`: `measurement_id,label[,group]`; `group` = unit fisik independen (aliquot/batch).
  Split train/test per group. Label hanya `[A-Za-z0-9_-]`, maks 12 karakter.

## Alur dengan dataset NYATA
```
python train_export.py --samples samples.csv --labels labels.csv \
    --kind trained --version v1 --out-rs ../../src/tinyml/model_params.rs \
    --report training_report.json --golden-out ../../host-tests/tests/fixtures/golden_data.rs
cd ../../host-tests && cargo +stable test --target <HOST_TRIPLE>
cd .. && cargo +esp build --release
```
Catatan: `--golden-out` pada dataset nyata hanya berisi 2 sesi uji; kasus tepi (`edge_*`)
hanya ada pada fixture sintetis dan hanya diperiksa bila ada.

## Fixture SINTETIK (yang disertakan)
`make_synthetic_fixture.py` membangkitkan data PALSU (RNG) -> model `synthetic_fixture` +
golden. Akurasinya (1.0) hanya bukti pipeline jalan, BUKAN klasifikasi aroma kopi.
