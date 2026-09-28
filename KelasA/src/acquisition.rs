//! `acquisition.rs`
//!
//! STEP 1 — SENSOR DATA (dummy): `SensorData` + `dummy_sensor_data()`.
//! STEP 2 — ACQUISITION (dummy, 1 Hz, 500 sample): `Sample` +
//! `run_dummy_acquisition_session()`.
//!
//! Modul ini mendefinisikan format data standar hasil pembacaan sensor
//! (`SensorData`) untuk sistem electronic nose, dan menjalankan sesi
//! sampling 1 Hz menggunakan sumber data dummy (`dummy_sensor_data`),
//! karena hardware (2x ADS1115, 8-channel gas sensor array, DHT22) belum
//! tersedia secara fisik.
//!
//! TIDAK ADA di modul ini pada tahap ini (sengaja):
//! - Pembacaan I2C/GPIO nyata ke ADS1115 atau DHT22.
//! - MQTT, Wi-Fi, ThingsBoard, TinyML, OTA, atau koneksi cloud apa pun.
//!
//! Ketika hardware sudah tersedia, `dummy_sensor_data()` di dalam
//! `run_dummy_acquisition_session()` akan diganti pembacaan sungguhan
//! (memanggil `drivers::ads1115` dan `drivers::dht22`), tanpa mengubah
//! bentuk `SensorData` maupun `Sample`.

/// Format data standar hasil satu siklus pembacaan sensor:
/// - 8 channel gas sensor (MQ3, MQ6, MQ7, MQ135, TGS2600, TGS2602, TGS2611, TGS2620)
/// - 1 nilai suhu (°C)
/// - 1 nilai kelembapan relatif (%RH)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SensorData {
    /// Nilai 8 channel gas sensor, indeks 0..7 = Gas 1..Gas 8.
    pub gas: [f32; 8],
    /// Suhu udara dalam derajat Celsius.
    pub temperature: f32,
    /// Kelembapan relatif dalam persen (%RH).
    pub humidity: f32,
}

/// Menghasilkan `SensorData` dengan nilai dummy tetap (bukan pembacaan
/// hardware nyata). Dipakai untuk memverifikasi bahwa struktur data,
/// alur pemanggilan, dan pencetakan ke serial monitor sudah benar,
/// sebelum driver ADS1115/DHT22 sungguhan dipasang.
pub fn dummy_sensor_data() -> SensorData {
    SensorData {
        gas: [1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90],
        temperature: 27.50,
        humidity: 65.00,
    }
}

// =====================================================================
// STEP 2 — ACQUISITION
// =====================================================================
//
// Menjalankan satu SESI pengukuran: sampling `SensorData` pada frekuensi
// 1 Hz selama `TOTAL_SAMPLES` detik, menggunakan sumber data dummy
// (`dummy_sensor_data`). Pembacaan ADS1115/DHT22 sungguhan BELUM ada di
// sini — akan menggantikan `dummy_sensor_data()` di step berikutnya tanpa
// mengubah bentuk `Sample`/`SensorData`.

use esp_hal::time::{Duration, Instant};

/// Jarak waktu antar sample (1 Hz = 1 sample setiap 1 detik).
pub const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// Jumlah sample dalam satu sesi pengukuran (durasi total 500 detik pada 1 Hz).
pub const TOTAL_SAMPLES: u32 = 500;

/// Satu hasil sampling: nomor urut sample + data sensor pada saat itu.
/// Nomor urut dipakai sebagai pengganti timestamp absolut — cukup untuk
/// tahap prototipe ini, dan gampang ditambah `Instant`/RTC nanti kalau perlu.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sample {
    pub number: u32,
    pub data: SensorData,
}

/// Menjalankan satu sesi akuisisi dummy: memanggil `on_sample` sebanyak
/// `TOTAL_SAMPLES` kali, satu kali per detik, lalu berhenti (return).
///
/// Timing pakai penjadwalan berbasis "titik waktu berikutnya"
/// (`next_tick += SAMPLE_INTERVAL`), BUKAN `delay(1 detik)` naif setelah
/// tiap sample. Alasannya: kalau `on_sample` (mis. proses cetak ke UART)
/// makan waktu beberapa milidetik, delay naif akan membuat interval
/// makin melenceng dari 1 detik seiring bertambahnya sample. Dengan
/// `next_tick` tetap, drift itu tidak terakumulasi.
///
/// Busy-wait (`while Instant::now() < next_tick {}`) dipakai karena project
/// ini belum pakai async runtime (embassy) — ini pendekatan standar & stabil
/// untuk prototipe single-threaded di atas `esp-hal` blocking API, sesuai
/// contoh resmi esp-hal (`Instant`/`Duration`), bukan `delay_ms()` sembarangan.
pub fn run_dummy_acquisition_session<F>(mut on_sample: F)
where
    F: FnMut(Sample),
{
    let mut next_tick = Instant::now();

    for number in 1..=TOTAL_SAMPLES {
        while Instant::now() < next_tick {}

        let sample = Sample {
            number,
            data: dummy_sensor_data(),
        };
        on_sample(sample);

        next_tick += SAMPLE_INTERVAL;
    }
}