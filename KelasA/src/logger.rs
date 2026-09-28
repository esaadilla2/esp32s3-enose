//! `logger.rs`
//!
//! STEP 3 — RAW DATA LOGGING (OFFLINE RAW DATASET PATH SAJA).
//!
//! Modul ini TIDAK melakukan smoothing/filtering/normalization/baseline
//! correction/feature extraction/averaging/PCA/ML preprocessing apa pun.
//! Setiap `Sample` dari `acquisition::run_dummy_acquisition_session()`
//! ditulis sebagai satu baris CSV, apa adanya, ke serial (host merekamnya
//! sebagai file .csv — lihat instruksi capture di bawah).
//!
//! TIDAK ADA di modul ini pada tahap ini (sengaja):
//! - SD card
//! - Wi-Fi / MQTT / ThingsBoard / cloud
//! - TinyML / Random Forest / feature extraction
//!
//! Field yang BELUM tersedia dari hardware sungguhan (raw ADC ADS1115,
//! voltage ADS1115, dht_age_s, timestamp wall-clock) diisi placeholder
//! yang ditandai jelas sebagai DUMMY/placeholder, BUKAN data sensor nyata.
//! Lihat komentar per-field di `log_sample()`.

use crate::acquisition::{Sample, SensorData};

/// ID sesi untuk tahap dummy/test saat ini. Ganti dengan skema
/// DAQ_ID/Batch_ID/Measurement_ID dsb. (lihat metadata project) begitu
/// desain metadata-session terpisah diimplementasikan.
pub const SESSION_ID: &str = "DUMMY-SESSION-0001";

/// Kontrak minimal "penulis CSV". Sengaja generik (bukan langsung
/// `esp_println::println!` di `main.rs`) supaya nanti tujuan penulisan bisa
/// diganti (mis. buffer, SD card) tanpa mengubah `main.rs`.
pub trait CsvLogger {
    /// Menulis baris header CSV. Harus dipanggil TEPAT SEKALI, sebelum
    /// baris data pertama.
    fn write_header(&mut self);

    /// Menulis satu baris CSV untuk satu `Sample`. Dipanggil TEPAT SEKALI
    /// per sample (satu baris = satu sample, urutan dijaga oleh
    /// `run_dummy_acquisition_session()`).
    fn log_sample(&mut self, session_id: &str, sample: Sample);

    /// Menandai akhir sesi setelah seluruh sample selesai ditulis.
    fn end_session(&mut self, session_id: &str, total_samples: u32);
}

/// Implementasi `CsvLogger` yang menulis ke serial (UART/USB-JTAG) lewat
/// `esp_println`. Host merekam output serial ini menjadi file `.csv`
/// (lihat instruksi "CARA MENJALANKAN" di penjelasan).
pub struct SerialCsvLogger;

impl SerialCsvLogger {
    pub fn new() -> Self {
        SerialCsvLogger
    }
}

impl CsvLogger for SerialCsvLogger {
    fn write_header(&mut self) {
        esp_println::println!(
            "t_s,timestamp,mq3_raw,mq6_raw,mq7_raw,mq135_raw,tgs2600_raw,tgs2602_raw,tgs2611_raw,tgs2620_raw,mq3_v,mq6_v,mq7_v,mq135_v,tgs2600_v,tgs2602_v,tgs2611_v,tgs2620_v,temp_c,rh_pct,dht_age_s"
        );
    }

    fn log_sample(&mut self, session_id: &str, sample: Sample) {
        // session_id belum dipakai di baris time-series (sesuai instruksi:
        // metadata session sebaiknya dipisah dari raw time-series, bukan
        // ditumpuk di tiap baris). Dipertahankan sebagai parameter untuk
        // dipakai nanti kalau desain metadata-terpisah diimplementasikan
        // (mis. ditulis sekali di baris/file metadata, bukan di sini).
        let _ = session_id;

        let SensorData {
            gas,
            temperature,
            humidity,
        } = sample.data;

        // t_s: REAL, bukan dummy — turunan langsung nomor urut sample yang
        // dijamin 1 Hz oleh run_dummy_acquisition_session() (bukan hasil
        // karangan logger.rs).
        let t_s = sample.number - 1;

        // timestamp: ESP32-S3 saat ini TIDAK punya RTC/NTP, jadi tidak ada
        // sumber wall-clock yang valid. Kolom ini diisi placeholder
        // eksplisit, BUKAN tanggal/jam karangan. Timestamp ISO8601
        // sebenarnya diisi HOST saat menerima baris ini via serial.
        let timestamp = "PENDING_HOST_TS";

        // mq*_raw / tgs*_raw: raw ADC count ADS1115. BELUM tersedia —
        // ADS1115 belum terpasang & dummy_sensor_data() tidak menghasilkan
        // nilai ADC. Diisi 0 sebagai placeholder DUMMY murni (sengaja TIDAK
        // diturunkan dari gas[], supaya tidak menyiratkan presisi/rentang
        // ADC yang belum tentu benar).
        let raw_placeholder: u16 = 0;

        // mq*_v / tgs*_v: voltage hasil ADS1115. BELUM tersedia — ADS1115
        // belum terpasang. gas[0..8] dari dummy_sensor_data() BUKAN hasil
        // pembacaan ADS1115 (lihat acquisition.rs, STEP 1), hanya angka
        // tetap untuk uji pipeline. Di sini gas[i] dipakai apa adanya untuk
        // mengisi kolom *_v agar bentuk/jumlah kolom CSV bisa diuji
        // end-to-end — TAPI ini tetap DUMMY, bukan voltage sensor nyata.
        // ASUMSI URUTAN (belum dikonfirmasi di kode): Gas1=MQ-3, Gas2=MQ-6,
        // Gas3=MQ-7, Gas4=MQ-135, Gas5=TGS2600, Gas6=TGS2602, Gas7=TGS2611,
        // Gas8=TGS2620, mengikuti urutan hardware pada dokumen project.
        let [mq3_v, mq6_v, mq7_v, mq135_v, tgs2600_v, tgs2602_v, tgs2611_v, tgs2620_v] = gas;

        // dht_age_s: umur pembacaan DHT22 terakhir (detik). BELUM dilacak
        // karena DHT22 belum terpasang & belum ada mekanisme "last read"
        // nyata. Diisi 0 sebagai placeholder DUMMY.
        let dht_age_s: u32 = 0;

        esp_println::println!(
            "{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{}",
            t_s,
            timestamp,
            raw_placeholder,
            raw_placeholder,
            raw_placeholder,
            raw_placeholder,
            raw_placeholder,
            raw_placeholder,
            raw_placeholder,
            raw_placeholder,
            mq3_v,
            mq6_v,
            mq7_v,
            mq135_v,
            tgs2600_v,
            tgs2602_v,
            tgs2611_v,
            tgs2620_v,
            temperature,
            humidity,
            dht_age_s,
        );
    }

    fn end_session(&mut self, session_id: &str, total_samples: u32) {
        // Baris penanda akhir sesi, diawali '#' supaya mudah diabaikan
        // (grep -v '^#') saat parsing CSV di tahap analisis nanti.
        esp_println::println!(
            "# END_SESSION session_id={} total_samples={}",
            session_id, total_samples
        );
    }
}