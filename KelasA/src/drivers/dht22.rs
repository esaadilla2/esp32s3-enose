//! DHT22 / AM2302 driver untuk ESP32-S3
//! DATA -> GPIO4
//!
//! Konfigurasi GPIO4 di atas berasal dari firmware yang sudah ada saat ini,
//! BUKAN asumsi baru -- tidak ada file referensi hardware yang menyebutkan
//! pin lain untuk pembacaan ini.
//!
//! PERBAIKAN VERSI INI (lihat penjelasan lengkap di luar file ini):
//! - Semua println!() DIHAPUS dari jalur timing-kritis (antara start signal
//!   sampai selesai baca 40 bit). esp_println menulis ke UART secara
//!   blocking dan bisa makan waktu ratusan us sampai beberapa ms per baris
//!   -- jauh lebih lama dari durasi total protokol DHT22 (~5ms untuk 40
//!   bit). Status fase kegagalan sekarang dilacak lewat Dht22Phase dan
//!   HANYA dicetak setelah read() gagal (di titik itu timing sudah selesai
//!   dipakai, jadi print tidak lagi mengganggu).
//! - Ditambahkan konfigurasi pull-up internal via InputConfig, karena
//!   sebelumnya tidak ada konfigurasi pull sama sekali (default kemungkinan
//!   floating).
//! - API Flex memakai set_output_enable()/set_input_enable() sesuai
//!   konfirmasi Anda (diverifikasi cocok dengan esp-hal 1.1.0).
//! - Tetap bit-bang manual, tetap Flex, tidak ada crate DHT22 eksternal,
//!   tidak ada dependency baru.
//!
//! CATATAN JUJUR: driver ini belum diuji di hardware fisik Anda. Perbaikan
//! di atas adalah hipotesis paling konsisten dengan log debug yang Anda
//! berikan (response LOW OK, response HIGH OK, gagal di bit pertama), tapi
//! belum ada jaminan ini pasti memperbaikinya sampai dites langsung.
//!
//! CATATAN INTERVAL PEMBACAAN: driver ini TIDAK mengatur jarak waktu antar
//! pemanggilan read(). Batas ~0.5 Hz (>=2 detik antar baca) DHT22 adalah
//! tanggung jawab pemanggil (acquisition_task) -- di luar cakupan file ini,
//! dan TIDAK diubah di sini sesuai instruksi Anda.

use esp_hal::delay::Delay;
use esp_hal::gpio::{Flex, InputConfig, Pull};
use esp_hal::time::{Duration, Instant};
use esp_println::println;

/// Timeout tiap fase protokol. 500us tetap dipertahankan -- generus
/// dibanding durasi normal tiap fase (~26-80us), dan sekarang benar-benar
/// mengukur waktu fase itu sendiri (bukan lagi ikut menghitung waktu
/// println! yang sudah dihapus dari jalur ini).
const PHASE_TIMEOUT: Duration = Duration::from_micros(500);

/// Ambang durasi fase "high" untuk membedakan bit 0 (~26-28us) dari bit 1
/// (~70us). 40us di tengah-tengah, sesuai instruksi Anda.
const BIT_THRESHOLD: Duration = Duration::from_micros(40);

/// Label fase protokol, dipakai HANYA untuk pelaporan error setelah gagal --
/// tidak pernah dicetak di tengah pembacaan yang sedang berjalan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dht22Phase {
    ResponseLow,
    ResponseHigh,
    FirstBitLow,
    BitHigh(u8),
    BitLow(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dht22Error {
    /// Pin tidak mencapai level yang diharapkan dalam PHASE_TIMEOUT, pada
    /// fase yang ditunjukkan.
    Timeout(Dht22Phase),
    ChecksumMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dht22Reading {
    pub temperature_c: f32,
    pub humidity_pct: f32,
}

pub struct Dht22<'d> {
    pin: Flex<'d>,
    delay: Delay,
}

impl<'d> Dht22<'d> {
    pub fn new(pin: Flex<'d>) -> Self {
        let mut pin = pin;
        // Pull-up internal -- shared antara mode input & output pada Flex
        // (lihat dokumentasi apply_input_config esp-hal 1.1.0). Dipasang
        // sekali di sini, bukan tiap read(), karena konfigurasi pull ini
        // persisten selama pin dipakai.
        pin.apply_input_config(&InputConfig::default().with_pull(Pull::Up));

        Self {
            pin,
            delay: Delay::new(),
        }
    }

    pub fn read(&mut self) -> Result<Dht22Reading, Dht22Error> {
        // ================================================================
        // 1. START SIGNAL -- boleh ada print di sini, di LUAR jalur
        //    timing-kritis (belum ada transisi cepat yang perlu ditangkap).
        // ================================================================
        self.pin.set_output_enable(true);
        self.pin.set_input_enable(false);

        self.pin.set_low();
        self.delay.delay_millis(2);

        self.pin.set_high();
        self.delay.delay_micros(30);

        self.pin.set_output_enable(false);
        self.pin.set_input_enable(true);

        // ================================================================
        // 2. RESPONSE DHT22 -- TIDAK ADA println! di antara langkah-langkah
        //    berikut sampai read() selesai/gagal.
        // ================================================================
        self.wait_for(false)
            .map_err(|_| Dht22Error::Timeout(Dht22Phase::ResponseLow))?;

        self.wait_for(true)
            .map_err(|_| Dht22Error::Timeout(Dht22Phase::ResponseHigh))?;

        // Transisi response HIGH -> LOW = awal bit pertama (~50us).
        self.wait_for(false)
            .map_err(|_| Dht22Error::Timeout(Dht22Phase::FirstBitLow))?;

        // ================================================================
        // 3. BACA 40 BIT
        // ================================================================
        let mut bytes = [0u8; 5];
        for (byte_index, byte) in bytes.iter_mut().enumerate() {
            for bit_index in 0..8u8 {
                let global_bit = (byte_index as u8) * 8 + bit_index;

                self.wait_for(true)
                    .map_err(|_| Dht22Error::Timeout(Dht22Phase::BitHigh(global_bit)))?;

                let high_start = Instant::now();

                self.wait_for(false)
                    .map_err(|_| Dht22Error::Timeout(Dht22Phase::BitLow(global_bit)))?;

                let high_duration = high_start.elapsed();

                *byte <<= 1;
                if high_duration > BIT_THRESHOLD {
                    *byte |= 1;
                }
            }
        }

        // ================================================================
        // 4. CHECKSUM -- data belum dianggap valid sebelum ini lolos.
        // ================================================================
        let checksum = bytes[0]
            .wrapping_add(bytes[1])
            .wrapping_add(bytes[2])
            .wrapping_add(bytes[3]);

        if checksum != bytes[4] {
            // Aman untuk print di sini -- pembacaan sudah selesai/gagal,
            // tidak ada lagi timing yang perlu dijaga.
            println!(
                "DHT22: checksum mismatch calculated={} received={}",
                checksum, bytes[4]
            );
            return Err(Dht22Error::ChecksumMismatch);
        }

        // ================================================================
        // 5. KONVERSI
        // ================================================================
        let humidity_raw = ((bytes[0] as u16) << 8) | (bytes[1] as u16);
        let humidity_pct = humidity_raw as f32 / 10.0;

        let temp_raw = (((bytes[2] & 0x7F) as u16) << 8) | (bytes[3] as u16);
        let mut temperature_c = temp_raw as f32 / 10.0;
        if bytes[2] & 0x80 != 0 {
            temperature_c = -temperature_c;
        }

        Ok(Dht22Reading {
            temperature_c,
            humidity_pct,
        })
    }

    fn wait_for(&mut self, want_high: bool) -> Result<(), ()> {
        let start = Instant::now();
        loop {
            let level_ok = if want_high {
                self.pin.is_high()
            } else {
                self.pin.is_low()
            };
            if level_ok {
                return Ok(());
            }
            if start.elapsed() > PHASE_TIMEOUT {
                return Err(());
            }
        }
    }
}