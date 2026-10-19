//! `acquisition.rs`
//!
//! Model data & orkestrasi akuisisi sesuai spesifikasi dataset terbaru:
//! MeasurementSession (warm-up -> measurement loop 1 Hz) menghasilkan
//! `MeasurementSample` per titik waktu, dipisah dari `MeasurementMetadata`
//! (satu per sesi).
//!
//! ASUMSI YANG BELUM DIKONFIRMASI (lihat juga penjelasan di luar file ini):
//! - Format "Coffee_ID" dalam Measurement_ID punya dua jenis prefix: 'C'
//!   (kopi) dan 'R' (referensi/ruang), disimpulkan dari contoh
//!   "DAQ03_B02_R00_A01_M01". BUKAN dari spesifikasi eksplisit.
//! - `QC_Flag` disimpan sebagai string bebas (bukan enum tertutup) karena
//!   nilai yang valid tidak disebutkan.
//!
//! MODE GAS
//! - `GasSource::Dummy`    : TIDAK ada transaksi I2C. Raw ADC dummy adalah
//!   angka tetap yang ditandai jelas, voltage dummy dihitung dari angka itu
//!   lewat `raw_to_voltage` yang SAMA dengan yang dipakai mode hardware
//!   (supaya bentuk datanya identik, sesuai permintaan).
//! - `GasSource::Hardware` : baca 2x ADS1115 di SATU bus I2C yang
//!   dipinjamkan (`&mut`) bergantian ke kedua `Ads1115`.
//!
//! MAPPING (jangan diubah urutannya; `gas[]` internal = urutan AIN per
//! chip, BUKAN urutan kolom CSV -- lihat `build_sample` untuk pemetaan
//! eksplisit ke nama kolom):
//!   ADS1115 @ 0x48: AIN0 -> MQ-6,    AIN1 -> MQ-135,
//!                   AIN2 -> MQ-7,    AIN3 -> MQ-3
//!   ADS1115 @ 0x49: AIN0 -> TGS2602, AIN1 -> TGS2620,
//!                   AIN2 -> TGS2611, AIN3 -> TGS2600
//!
//! DHT22: primary ("chamber") masuk `MeasurementSample`. Ambient (opsional)
//! hanya dibaca sebagai cadangan info saat primary gagal, tidak masuk
//! sample. `dht_age_s` = detik sejak pembacaan primary valid terakhir;
//! kalau belum pernah berhasil, usianya = seluruh waktu berjalan sejak
//! awal measurement (t_s itu sendiri).
//!
//! TIMESTAMP: ESP32 belum punya RTC/NTP. `timestamp` diisi
//! `TIMESTAMP_PLACEHOLDER`, BUKAN waktu dunia nyata -- diisi ulang di host
//! saat capture (lihat instruksi capture CSV sebelumnya di project ini).
//! `t_s` (elapsed detik sejak awal measurement, bukan sejak boot) tetap
//! akurat tanpa RTC.
//!
//! STEP 1 TELEMETRY: `DataMode` + `Acquisition::data_mode()` ditambahkan
//! supaya payload MQTT (dibangun di main.rs) bisa menandai data_mode
//! berdasarkan `GasSource` yang BENAR-BENAR aktif pada instance ini,
//! bukan ditebak di sisi mqtt_task.

use core::fmt::Write as _;

use embassy_sync::blocking_mutex::raw::RawMutex;
use embassy_sync::channel::Channel;
use embassy_time::{Duration, Instant, Timer};
use embedded_hal::i2c::I2c as I2cBus;
use esp_println::println;
use heapless::String as HString;

use crate::drivers::ads1115::{Ads1115, Channel as AdcChannel, DEFAULT_PGA, Pga, raw_to_voltage};
use crate::drivers::dht22::Dht22;
use crate::tinyml::features::{
    DHT_MAX_AGE_S, FeatureAccumulator, FeatureError, FeatureFrame, FeatureVector,
};

// =====================================================================
// KONFIGURASI SESI (Heating_s, Duration_s, Fs_Hz, PGA, Divider_Ratio)
// =====================================================================

/// DHT22 dibaca tiap N detik (interval 1 Hz, jadi N=2 -> tiap 2 detik),
/// mematuhi batas ~0,5 Hz sensor yang sudah ditetapkan sebelumnya di
/// project ini. Di detik lain, nilai terakhir di-hold.
pub const DHT_READ_EVERY_N_SECONDS: u32 = 2;

#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    pub heating_s: u32,
    pub duration_s: u32,
    pub fs_hz: f32,
    /// Harus sinkron dengan PGA yang dipakai `drivers::ads1115`
    /// (`CFG_PGA_4_096V` -> 4.096). Dicatat di sini hanya untuk metadata,
    /// TIDAK dipakai untuk kalkulasi apa pun di modul ini.
    pub pga_fsr_v: f32,
    /// `None` = belum diketahui. JANGAN diisi angka karangan -- rasio
    /// pembagi tegangan hardware belum ditentukan sampai Anda konfirmasi.
    pub divider_ratio: Option<f32>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            heating_s: 0,
            duration_s: 300,
            fs_hz: 1.0,
            pga_fsr_v: 4.096,
            divider_ratio: None,
        }
    }
}

/// Error validasi `SessionConfig`. Tidak ada varian untuk `heating_s`
/// karena tidak ada batas teknis yang berlaku di level modul ini -- 0
/// (tanpa warm-up) valid, dan tidak ada batas atas yang bisa dijustifikasi
/// tanpa data hardware/eksperimen tambahan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionConfigError {
    /// `fs_hz` <= 0.0, NaN, atau infinite -- tidak bisa jadi pembagi
    /// untuk menghitung period sampling.
    InvalidFsHz,
    /// `fs_hz` > 1000.0. Ini BUKAN batas sembarangan: period_ms dibulatkan
    /// ke milidetik (`(1000.0 / fs_hz).round()`), jadi fs_hz > 1000.0
    /// menghasilkan period_ms < 1, yang berarti `Timer::at` dipanggil
    /// dengan deadline yang sama berulang kali -- busy-loop, bukan
    /// sampling nyata.
    FsHzTooHigh,
    /// Hasil `1000.0 / fs_hz` melebihi `u64::MAX` setelah dibulatkan.
    /// Ini kasus ekor fs_hz yang sangat kecil tapi masih > 0.0 (mis.
    /// fs_hz = 1e-30) -- secara teknis "valid" terhadap pengecekan di
    /// atas tapi akan membuat `as u64` saturating ke `u64::MAX` secara
    /// diam-diam. Ditolak eksplisit di sini supaya gagalnya kelihatan,
    /// bukan diam-diam salah.
    PeriodOverflow,
    /// `duration_s == 0` -- sesi tanpa sample tidak punya arti.
    InvalidDurationS,
    /// `pga_fsr_v` <= 0.0, NaN, atau infinite -- tidak bisa
    /// merepresentasikan full-scale range ADC yang valid, walau field
    /// ini cuma dipakai untuk metadata/audit, bukan kalkulasi.
    InvalidPgaFsrV,
    /// `divider_ratio` ada isinya (`Some`) tapi <= 0.0, NaN, atau
    /// infinite. `None` (belum diketahui) tetap valid.
    InvalidDividerRatio,
}

impl SessionConfig {
    /// Validasi semua parameter. Lihat `SessionConfigError` untuk alasan
    /// tiap batas.
    pub fn validate(&self) -> Result<(), SessionConfigError> {
        if !self.fs_hz.is_finite() || self.fs_hz <= 0.0 {
            return Err(SessionConfigError::InvalidFsHz);
        }
        if self.fs_hz > 1000.0 {
            return Err(SessionConfigError::FsHzTooHigh);
        }
        if self.duration_s == 0 {
            return Err(SessionConfigError::InvalidDurationS);
        }
        if !self.pga_fsr_v.is_finite() || self.pga_fsr_v <= 0.0 {
            return Err(SessionConfigError::InvalidPgaFsrV);
        }
        if let Some(ratio) = self.divider_ratio
            && (!ratio.is_finite() || ratio <= 0.0)
        {
            return Err(SessionConfigError::InvalidDividerRatio);
        }
        Ok(())
    }

    /// Hitung period sampling dalam milidetik dari `fs_hz`, sudah lewat
    /// `validate()`. Tidak pernah mengembalikan 0, NaN, infinite, atau
    /// hasil overflow diam-diam -- semua kasus itu jadi `Err` eksplisit.
    pub fn sample_period_ms(&self) -> Result<u64, SessionConfigError> {
        self.validate()?;

        let period_f = 1000.0_f64 / self.fs_hz as f64;

        if !period_f.is_finite() || period_f < 1.0 || period_f >= u64::MAX as f64 {
            return Err(SessionConfigError::PeriodOverflow);
        }

        Ok((period_f + 0.5) as u64)
    }
}

// =====================================================================
// IDENTITAS: Measurement_ID & hierarki
// =====================================================================

pub const MEASUREMENT_ID_CAPACITY: usize = 32;
pub type MeasurementIdString = HString<MEASUREMENT_ID_CAPACITY>;

/// Jenis Coffee_ID pada pola Measurement_ID: 'C' (kopi) atau 'R'
/// (referensi/ruang, contoh "R00"). ASUMSI -- lihat catatan di atas file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoffeeIdKind {
    Coffee,
    Reference,
}

impl CoffeeIdKind {
    fn prefix(self) -> char {
        match self {
            CoffeeIdKind::Coffee => 'C',
            CoffeeIdKind::Reference => 'R',
        }
    }
}

/// Bangun Measurement_ID sesuai pola DAQnn_Bnn_[C|R]nn_Ann_Mnn, mis.
/// "DAQ03_B02_C07_A04_M04". nn di-zero-pad 2 digit; nilai >99 tetap
/// dicetak apa adanya (tidak dipotong), hanya lebih dari 2 digit.
pub fn build_measurement_id(
    daq_id: u8,
    batch_id: u8,
    coffee_kind: CoffeeIdKind,
    coffee_id: u8,
    aliquot_id: u8,
    measurement_no: u8,
) -> MeasurementIdString {
    let mut s = MeasurementIdString::new();
    let _ = write!(
        s,
        "DAQ{:02}_B{:02}_{}{:02}_A{:02}_M{:02}",
        daq_id,
        batch_id,
        coffee_kind.prefix(),
        coffee_id,
        aliquot_id,
        measurement_no
    );
    s
}

// =====================================================================
// METADATA SESI (satu per sesi, TIDAK diulang per sample)
// =====================================================================

/// Placeholder timestamp karena ESP32 belum punya RTC/NTP. BUKAN waktu
/// dunia nyata -- diisi ulang di host saat capture.
pub const TIMESTAMP_PLACEHOLDER: &str = "PENDING_HOST_TS";

#[derive(Debug, Clone)]
pub struct MeasurementMetadata {
    pub measurement_id: MeasurementIdString,
    pub daq_id: u8,
    pub batch_id: u8,
    pub group_id: HString<16>,
    pub coffee_id: HString<16>,
    pub species: HString<32>,
    pub bean_state: HString<32>,
    pub aliquot_id: u8,
    /// `None` = belum ditimbang / belum diketahui.
    pub mass_g: Option<f32>,
    pub timestamp: HString<32>,
    /// Suhu ruang saat metadata dicatat (BUKAN dari DHT22 acquisition loop).
    pub temperature: Option<f32>,
    pub humidity: Option<f32>,
    /// Format bebas sesuai lembar data eksperimen Anda -- TIDAK di-parse.
    pub room_temp_room_rh: HString<32>,
    pub warmup_min: f32,
    pub interval_min: f32,
    pub run_order: u32,
    pub heating_s: u32,
    pub duration_s: u32,
    pub fs_hz: f32,
    pub pga_fsr_v: f32,
    pub divider_ratio: Option<f32>,
    pub protocol_ver: HString<16>,
    pub operator: HString<32>,
    /// String bebas -- lihat catatan ASUMSI di atas file.
    pub qc_flag: HString<16>,
    pub notes: HString<64>,
}

// =====================================================================
// SAMPLE (satu per titik waktu). Urutan field SENGAJA mengikuti urutan
// kolom CSV target Anda persis, supaya tidak ada ambiguitas urutan.
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementSample {
    pub t_s: u32,
    pub timestamp: HString<32>,
    pub mq3_raw: i16,
    pub mq6_raw: i16,
    pub mq7_raw: i16,
    pub mq135_raw: i16,
    pub tgs2600_raw: i16,
    pub tgs2602_raw: i16,
    pub tgs2611_raw: i16,
    pub tgs2620_raw: i16,
    pub mq3_v: f32,
    pub mq6_v: f32,
    pub mq7_v: f32,
    pub mq135_v: f32,
    pub tgs2600_v: f32,
    pub tgs2602_v: f32,
    pub tgs2611_v: f32,
    pub tgs2620_v: f32,
    pub temp_c: f32,
    pub rh_pct: f32,
    pub dht_age_s: u32,
    /// Bitmask status pembacaan 8 sensor gas pada sampel saat ini:
    /// bit 0: MQ-6 (0x48 Ain0)
    /// bit 1: MQ-135 (0x48 Ain1)
    /// bit 2: MQ-7 (0x48 Ain2)
    /// bit 3: MQ-3 (0x48 Ain3)
    /// bit 4: TGS2602 (0x49 Ain0)
    /// bit 5: TGS2620 (0x49 Ain1)
    /// bit 6: TGS2611 (0x49 Ain2)
    /// bit 7: TGS2600 (0x49 Ain3)
    /// Nilai 0xFF (255) = semua 8 sensor gas berhasil dibaca segar (fresh).
    pub gas_ok_mask: u8,
}

// =====================================================================
// PEMETAAN SENSOR GAS DAN SUMBER DATA
// =====================================================================

/// Pemetaan fisik kanal ADS1115 ke sensor gas:
/// Indeks 0..3 pada ADS1115 unit #1 (0x48): MQ-6, MQ-135, MQ-7, MQ-3
/// Indeks 4..7 pada ADS1115 unit #2 (0x49): TGS2602, TGS2620, TGS2611, TGS2600
pub const GAS_CHANNEL_MAPPING: [(&str, u8, AdcChannel); 8] = [
    ("MQ-6", 0x48, AdcChannel::Ain0),
    ("MQ-135", 0x48, AdcChannel::Ain1),
    ("MQ-7", 0x48, AdcChannel::Ain2),
    ("MQ-3", 0x48, AdcChannel::Ain3),
    ("TGS2602", 0x49, AdcChannel::Ain0),
    ("TGS2620", 0x49, AdcChannel::Ain1),
    ("TGS2611", 0x49, AdcChannel::Ain2),
    ("TGS2600", 0x49, AdcChannel::Ain3),
];

pub enum GasSource<I2C> {
    /// Tanpa I2C. Raw ADC dummy, BUKAN hasil ADC fisik.
    Dummy,
    Hardware {
        i2c: I2C,
        adc_first: Ads1115,
        adc_second: Ads1115,
    },
}

impl<I2C> GasSource<I2C> {
    /// Buat GasSource mode hardware dengan PGA default (DEFAULT_PGA = +-4.096V).
    pub fn hardware(i2c: I2C, addr_first: u8, addr_second: u8) -> Self {
        Self::hardware_with_pga(i2c, addr_first, addr_second, DEFAULT_PGA)
    }

    /// Buat GasSource mode hardware dengan PGA yang ditentukan secara eksplisit.
    pub fn hardware_with_pga(i2c: I2C, addr_first: u8, addr_second: u8, pga: Pga) -> Self {
        GasSource::Hardware {
            i2c,
            adc_first: Ads1115::new(addr_first).with_pga(pga),
            adc_second: Ads1115::new(addr_second).with_pga(pga),
        }
    }
}

/// Sumber data gas yang BENAR-BENAR aktif untuk instance ini. Dipakai
/// supaya `data_mode` pada telemetry MQTT (lihat main.rs) ditentukan dari
/// `GasSource` sesungguhnya, bukan ditebak di sisi mqtt_task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataMode {
    Dummy,
    Hardware,
}

impl DataMode {
    pub fn as_str(self) -> &'static str {
        match self {
            DataMode::Dummy => "dummy",
            DataMode::Hardware => "hardware",
        }
    }
}

/// Nilai raw ADC dummy tetap, satu per channel (urutan AIN internal,
/// lihat komentar mapping di atas file). BUKAN hasil ADC fisik.
const DUMMY_GAS_RAW: [i16; 8] = [12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700];

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GasChannelState {
    pub last_raw: i16,
    pub last_voltage: f32,
    pub ever_ok: bool,
}

fn hold_note(ever_ok: bool) -> &'static str {
    if ever_ok {
        "pakai nilai terakhir yang valid"
    } else {
        "belum ada pembacaan valid, pakai placeholder 0"
    }
}

// =====================================================================
// ACQUISITION
// =====================================================================

pub struct Acquisition<I2C> {
    gas_source: GasSource<I2C>,
    /// DHT22 primary (chamber). `None` = suhu/RH dummy.
    dht22: Option<Dht22<'static>>,
    /// DHT22 ambient, hanya dibaca saat primary gagal. Tidak masuk sample.
    dht22_ambient: Option<Dht22<'static>>,
    /// Indeks 0..7 = urutan AIN (lihat mapping di atas file), BUKAN
    /// urutan kolom CSV.
    gas: [GasChannelState; 8],
    last_gas_ok_mask: u8,
    last_temperature: f32,
    last_humidity: f32,
    last_ambient_temperature: f32,
    last_ambient_humidity: f32,
    dht_last_success_t_s: Option<u32>,
    ambient_ever_ok: bool,
    /// Akumulator fitur TinyML per sesi (streaming, RAM tetap ~672 byte).
    features: FeatureAccumulator,
}

impl<I2C> Acquisition<I2C> {
    pub fn new(gas_source: GasSource<I2C>, dht22: Option<Dht22<'static>>) -> Self {
        Self {
            gas_source,
            dht22,
            dht22_ambient: None,
            gas: [GasChannelState::default(); 8],
            last_gas_ok_mask: 0xFF,
            last_temperature: 0.0,
            last_humidity: 0.0,
            last_ambient_temperature: 0.0,
            last_ambient_humidity: 0.0,
            dht_last_success_t_s: None,
            ambient_ever_ok: false,
            features: FeatureAccumulator::new(),
        }
    }

    /// Pasang DHT22 ambient sebagai cadangan (dibaca hanya saat primary gagal).
    pub fn with_ambient_backup(mut self, dht22_ambient: Dht22<'static>) -> Self {
        self.dht22_ambient = Some(dht22_ambient);
        self
    }

    /// Mode data gas yang benar-benar aktif pada instance ini. Dipakai
    /// untuk menandai `data_mode` pada telemetry MQTT.
    pub fn data_mode(&self) -> DataMode {
        match self.gas_source {
            GasSource::Dummy => DataMode::Dummy,
            GasSource::Hardware { .. } => DataMode::Hardware,
        }
    }

    /// Fitur TinyML (58) dari sesi yang baru selesai. `Err` jika data valid
    /// belum memenuhi syarat minimum -- JANGAN diganti nilai pengganti.
    pub fn session_features(&self) -> Result<FeatureVector, FeatureError> {
        self.features.finalize()
    }

    /// Jumlah frame yang disuap ke akumulator fitur pada sesi terakhir.
    pub fn session_frames(&self) -> u32 {
        self.features.frames()
    }
}

impl<I2C> Acquisition<I2C>
where
    I2C: I2cBus,
{
    /// Jalankan satu sesi: warm-up (`config.heating_s`), lalu measurement
    /// loop 1 Hz (`config.duration_s` titik, t_s = 0..duration_s-1), lalu
    /// return. Timing memakai deadline tetap (`next_tick += period`) +
    /// `Timer::at`, BUKAN `Timer::after` berulang, supaya lama pembacaan
    /// sensor tidak menggeser jadwal. Kegagalan sensor TIDAK menghentikan
    /// sesi.
    pub async fn run_session<M, const N: usize>(
        &mut self,
        metadata: &MeasurementMetadata,
        config: SessionConfig,
        channel: &Channel<M, MeasurementSample, N>,
    ) -> Result<(), SessionConfigError>
    where
        M: RawMutex,
    {
        // Validasi dulu, SEBELUM log "SESSION START" -- kalau config tidak
        // valid, sesi dianggap tidak pernah dimulai, bukan "dimulai lalu
        // keliru di tengah jalan".
        let period_ms = config.sample_period_ms()?;

        // Fitur TinyML dihitung per sesi: mulai dari nol.
        self.features.reset();

        println!();
        println!("=== MEASUREMENT SESSION START ===");
        println!("Measurement_ID: {}", metadata.measurement_id);
        match &self.gas_source {
            GasSource::Dummy => println!("Mode gas: DUMMY -- raw/voltage BUKAN hasil ADC fisik"),
            GasSource::Hardware { .. } => println!("Mode gas: ADS1115 (0x48 + 0x49)"),
        }
        match &self.dht22 {
            None => println!("Mode suhu/RH: DUMMY -- BUKAN hasil DHT22 fisik"),
            Some(_) => println!("Mode suhu/RH: DHT22 fisik (primary/chamber)"),
        }

        if config.heating_s > 0 {
            println!("Pemanasan internal {} s dimulai...", config.heating_s);
            Timer::after(Duration::from_secs(config.heating_s as u64)).await;
        } else {
            println!(
                "heating_s=0: tidak ada pemanasan internal (sensor dipanaskan di luar firmware)."
            );
        }
        println!(
            "Mulai measurement ({} s, {} Hz, period {} ms).",
            config.duration_s, config.fs_hz, period_ms
        );

        let period = Duration::from_millis(period_ms);
        let mut next_tick = Instant::now();
        let mut dropped: u32 = 0;
        let mut resyncs: u32 = 0;

        for t_s in 0..config.duration_s {
            Timer::at(next_tick).await;

            self.update_environment(t_s);
            self.update_gas(t_s);

            let sample = self.build_sample(t_s);

            // TinyML: suap akumulator SEBELUM try_send, supaya sample yang
            // dibuang karena channel penuh tetap masuk fitur. O(8) operasi.
            let frame = self.build_frame(&sample);
            self.features.push(&frame);

            println!(
                "t_s={}, gas_ok=0x{:02X}, mq6_raw={}, mq135_raw={}, mq7_raw={}, mq3_raw={}, tgs2602_raw={}, tgs2620_raw={}, tgs2611_raw={}, tgs2600_raw={}, temp_c={:.2}, rh_pct={:.2}, dht_age_s={}",
                sample.t_s,
                sample.gas_ok_mask,
                sample.mq6_raw,
                sample.mq135_raw,
                sample.mq7_raw,
                sample.mq3_raw,
                sample.tgs2602_raw,
                sample.tgs2620_raw,
                sample.tgs2611_raw,
                sample.tgs2600_raw,
                sample.temp_c,
                sample.rh_pct,
                sample.dht_age_s,
            );

            // try_send: akuisisi TIDAK PERNAH menunggu MQTT. Kalau channel
            // penuh, sample dibuang dan dihitung (celah t_s terlihat di host).
            if channel.try_send(sample).is_err() {
                dropped += 1;
                println!(
                    "CHANNEL PENUH: sample t_s={} dibuang (total dibuang={})",
                    t_s, dropped
                );
            }

            next_tick += period;

            // Anti catch-up burst: kalau sudah telat >= 1 periode (mis. sensor
            // atau println tersendat), geser jadwal ke sekarang.
            let now = Instant::now();
            if now >= next_tick + period {
                resyncs += 1;
                println!(
                    "TIMING TELAT di t_s={}: jadwal digeser, bukan burst (resync={})",
                    t_s, resyncs
                );
                next_tick = now;
            }
        }

        println!(
            "=== MEASUREMENT SESSION FINISHED (dibuang={}, resync={}) ===",
            dropped, resyncs
        );
        Ok(())
    }

    fn build_sample(&self, t_s: u32) -> MeasurementSample {
        let mut timestamp: HString<32> = HString::new();
        let _ = timestamp.push_str(TIMESTAMP_PLACEHOLDER);

        // Pemetaan eksplisit indeks AIN internal -> nama kolom CSV:
        // gas[0]=MQ-6 (0x48 A0), gas[1]=MQ-135 (0x48 A1), gas[2]=MQ-7 (0x48 A2), gas[3]=MQ-3 (0x48 A3),
        // gas[4]=TGS2602 (0x49 A0), gas[5]=TGS2620 (0x49 A1), gas[6]=TGS2611 (0x49 A2), gas[7]=TGS2600 (0x49 A3)
        MeasurementSample {
            t_s,
            timestamp,
            mq3_raw: self.gas[3].last_raw,
            mq6_raw: self.gas[0].last_raw,
            mq7_raw: self.gas[2].last_raw,
            mq135_raw: self.gas[1].last_raw,
            tgs2600_raw: self.gas[7].last_raw,
            tgs2602_raw: self.gas[4].last_raw,
            tgs2611_raw: self.gas[6].last_raw,
            tgs2620_raw: self.gas[5].last_raw,
            mq3_v: self.gas[3].last_voltage,
            mq6_v: self.gas[0].last_voltage,
            mq7_v: self.gas[2].last_voltage,
            mq135_v: self.gas[1].last_voltage,
            tgs2600_v: self.gas[7].last_voltage,
            tgs2602_v: self.gas[4].last_voltage,
            tgs2611_v: self.gas[6].last_voltage,
            tgs2620_v: self.gas[5].last_voltage,
            temp_c: self.last_temperature,
            rh_pct: self.last_humidity,
            dht_age_s: self.current_dht_age_s(t_s),
            gas_ok_mask: self.last_gas_ok_mask,
        }
    }

    /// Bentuk `FeatureFrame` dari sample. URUTAN gas1..gas8 = urutan payload MQTT
    /// (gas1=MQ-6, gas2=MQ-135, gas3=MQ-7, gas4=MQ-3, gas5=TGS2602, gas6=TGS2620,
    /// gas7=TGS2611, gas8=TGS2600) = indeks gas[0..7] = bit 0..7 `gas_ok_mask`.
    /// Kalau urutan ini berubah, model TinyML lama TIDAK valid lagi.
    fn build_frame(&self, s: &MeasurementSample) -> FeatureFrame {
        let env_valid = match &self.dht22 {
            // Mode dummy suhu/RH: uji integrasi saja (tanpa sensor).
            None => true,
            Some(_) => s.dht_age_s <= DHT_MAX_AGE_S,
        };
        FeatureFrame {
            t_s: s.t_s,
            gas_v: [
                s.mq6_v, s.mq135_v, s.mq7_v, s.mq3_v, s.tgs2602_v, s.tgs2620_v, s.tgs2611_v,
                s.tgs2600_v,
            ],
            gas_valid: core::array::from_fn(|i| (s.gas_ok_mask >> i) & 1 == 1),
            temp_c: s.temp_c,
            rh_pct: s.rh_pct,
            env_valid,
        }
    }

    fn current_dht_age_s(&self, t_s: u32) -> u32 {
        match self.dht_last_success_t_s {
            Some(last) => t_s.saturating_sub(last),
            // Belum pernah berhasil -- usia = seluruh waktu berjalan sejak
            // awal measurement.
            None => t_s,
        }
    }

    fn update_gas(&mut self, t_s: u32) {
        match &mut self.gas_source {
            GasSource::Dummy => {
                for (i, &raw) in DUMMY_GAS_RAW.iter().enumerate() {
                    self.gas[i].last_raw = raw;
                    self.gas[i].last_voltage = raw_to_voltage(raw);
                    self.gas[i].ever_ok = true;
                }
                self.last_gas_ok_mask = 0xFF;
            }
            GasSource::Hardware {
                i2c,
                adc_first,
                adc_second,
            } => {
                let plan: [(usize, &Ads1115, AdcChannel); 8] = [
                    (0, &*adc_first, AdcChannel::Ain0),  // MQ-6
                    (1, &*adc_first, AdcChannel::Ain1),  // MQ-135
                    (2, &*adc_first, AdcChannel::Ain2),  // MQ-7
                    (3, &*adc_first, AdcChannel::Ain3),  // MQ-3
                    (4, &*adc_second, AdcChannel::Ain0), // TGS2602
                    (5, &*adc_second, AdcChannel::Ain1), // TGS2620
                    (6, &*adc_second, AdcChannel::Ain2), // TGS2611
                    (7, &*adc_second, AdcChannel::Ain3), // TGS2600
                ];

                let mut current_ok_mask: u8 = 0;
                for (idx, adc, ain) in plan {
                    match adc.read_channel_raw(&mut *i2c, ain) {
                        Ok(raw) => {
                            self.gas[idx].last_raw = raw;
                            self.gas[idx].last_voltage = adc.pga().raw_to_voltage(raw);
                            self.gas[idx].ever_ok = true;
                            current_ok_mask |= 1 << idx;
                        }
                        Err(e) => println!(
                            "ADS1115 0x{:02X} channel {:?} gagal (t_s={}): {:?} -- {}",
                            adc.address(),
                            ain,
                            t_s,
                            e,
                            hold_note(self.gas[idx].ever_ok)
                        ),
                    }
                }
                self.last_gas_ok_mask = current_ok_mask;
            }
        }
    }

    fn update_environment(&mut self, t_s: u32) {
        match &mut self.dht22 {
            None => {
                // Mode dummy: suhu/RH tetap, bukan hasil DHT22 fisik.
                self.last_temperature = 27.50;
                self.last_humidity = 65.00;
            }
            Some(chamber) => {
                if t_s % DHT_READ_EVERY_N_SECONDS == 0 {
                    match chamber.read() {
                        Ok(reading) => {
                            self.last_temperature = reading.temperature_c;
                            self.last_humidity = reading.humidity_pct;
                            self.dht_last_success_t_s = Some(t_s);
                        }
                        Err(e) => {
                            println!(
                                "DHT22 CHAMBER gagal (t_s={}): {:?} -- nilai terakhir dipertahankan",
                                t_s, e
                            );

                            if let Some(ambient) = &mut self.dht22_ambient {
                                match ambient.read() {
                                    Ok(r) => {
                                        self.last_ambient_temperature = r.temperature_c;
                                        self.last_ambient_humidity = r.humidity_pct;
                                        self.ambient_ever_ok = true;
                                        println!(
                                            "DHT22 AMBIENT (info saja, tidak masuk sample): temperature={:.2} C, humidity={:.2} %",
                                            r.temperature_c, r.humidity_pct
                                        );
                                    }
                                    Err(e2) => println!(
                                        "DHT22 AMBIENT gagal (t_s={}): {:?} -- {}",
                                        t_s,
                                        e2,
                                        hold_note(self.ambient_ever_ok)
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// =====================================================================
// JSON PAYLOAD FORMATTER (THINGSBOARD TELEMETRY)
// =====================================================================

/// Bangun payload JSON telemetri ThingsBoard dari `MeasurementSample`.
/// Memetakan 8 gas sesuai urutan fisik kanal ADC:
///   gas1 = MQ-6   (0x48 Ain0)
///   gas2 = MQ-135 (0x48 Ain1)
///   gas3 = MQ-7   (0x48 Ain2)
///   gas4 = MQ-3   (0x48 Ain3)
///   gas5 = TGS2602 (0x49 Ain0)
///   gas6 = TGS2620 (0x49 Ain1)
///   gas7 = TGS2611 (0x49 Ain2)
///   gas8 = TGS2600 (0x49 Ain3)
pub fn build_json_payload<const CAP: usize>(
    sample: &MeasurementSample,
    measurement_id: &str,
    data_mode: &str,
    out: &mut HString<CAP>,
) -> Result<(), core::fmt::Error> {
    out.clear();

    write!(
        out,
        "{{\"measurement_id\":\"{}\",\"sample\":{},\"gas1\":{:.3},\"gas2\":{:.3},\"gas3\":{:.3},\"gas4\":{:.3},\"gas5\":{:.3},\"gas6\":{:.3},\"gas7\":{:.3},\"gas8\":{:.3},\"temperature\":{:.2},\"humidity\":{:.2},\"data_mode\":\"{}\",\"gas_ok\":{}}}",
        measurement_id,
        sample.t_s,
        sample.mq6_v,
        sample.mq135_v,
        sample.mq7_v,
        sample.mq3_v,
        sample.tgs2602_v,
        sample.tgs2620_v,
        sample.tgs2611_v,
        sample.tgs2600_v,
        sample.temp_c,
        sample.rh_pct,
        data_mode,
        sample.gas_ok_mask,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fs_hz_zero_is_rejected() {
        let config = SessionConfig {
            fs_hz: 0.0,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidFsHz));
    }

    #[test]
    fn fs_hz_negative_is_rejected() {
        let config = SessionConfig {
            fs_hz: -1.0,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidFsHz));
    }

    #[test]
    fn fs_hz_nan_is_rejected() {
        let config = SessionConfig {
            fs_hz: f32::NAN,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidFsHz));
    }

    #[test]
    fn fs_hz_infinite_is_rejected() {
        let config = SessionConfig {
            fs_hz: f32::INFINITY,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidFsHz));
    }

    #[test]
    fn fs_hz_too_high_is_rejected() {
        let config = SessionConfig {
            fs_hz: 1000.1,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::FsHzTooHigh));
    }

    #[test]
    fn fs_hz_at_upper_boundary_is_accepted() {
        let config = SessionConfig {
            fs_hz: 1000.0,
            ..SessionConfig::default()
        };
        assert_eq!(config.sample_period_ms(), Ok(1));
    }

    #[test]
    fn default_fs_hz_gives_1000ms_period() {
        let config = SessionConfig::default();
        assert_eq!(config.sample_period_ms(), Ok(1000));
    }

    #[test]
    fn duration_zero_is_rejected() {
        let config = SessionConfig {
            duration_s: 0,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidDurationS));
    }

    #[test]
    fn pga_fsr_zero_is_rejected() {
        let config = SessionConfig {
            pga_fsr_v: 0.0,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidPgaFsrV));
    }

    #[test]
    fn pga_fsr_negative_is_rejected() {
        let config = SessionConfig {
            pga_fsr_v: -4.096,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidPgaFsrV));
    }

    #[test]
    fn divider_ratio_none_is_accepted() {
        let config = SessionConfig {
            divider_ratio: None,
            ..SessionConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn divider_ratio_zero_is_rejected() {
        let config = SessionConfig {
            divider_ratio: Some(0.0),
            ..SessionConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(SessionConfigError::InvalidDividerRatio)
        );
    }

    #[test]
    fn divider_ratio_positive_is_accepted() {
        let config = SessionConfig {
            divider_ratio: Some(2.0),
            ..SessionConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn heating_zero_is_accepted() {
        let config = SessionConfig {
            heating_s: 0,
            ..SessionConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pga_fsr_nan_is_rejected() {
        let config = SessionConfig {
            pga_fsr_v: f32::NAN,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidPgaFsrV));
    }

    #[test]
    fn pga_fsr_infinite_is_rejected() {
        let config = SessionConfig {
            pga_fsr_v: f32::INFINITY,
            ..SessionConfig::default()
        };
        assert_eq!(config.validate(), Err(SessionConfigError::InvalidPgaFsrV));
    }

    #[test]
    fn divider_ratio_negative_is_rejected() {
        let config = SessionConfig {
            divider_ratio: Some(-2.0),
            ..SessionConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(SessionConfigError::InvalidDividerRatio)
        );
    }

    #[test]
    fn divider_ratio_nan_is_rejected() {
        let config = SessionConfig {
            divider_ratio: Some(f32::NAN),
            ..SessionConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(SessionConfigError::InvalidDividerRatio)
        );
    }

    #[test]
    fn divider_ratio_infinite_is_rejected() {
        let config = SessionConfig {
            divider_ratio: Some(f32::INFINITY),
            ..SessionConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(SessionConfigError::InvalidDividerRatio)
        );
    }

    #[test]
    fn production_config_step2_is_valid_with_1hz_period() {
        let config = SessionConfig {
            heating_s: 0,
            duration_s: 300,
            fs_hz: 1.0,
            ..SessionConfig::default()
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.sample_period_ms(), Ok(1000));
    }
}
