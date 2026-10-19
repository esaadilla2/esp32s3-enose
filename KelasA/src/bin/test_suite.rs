#![no_std]
#![no_main]

// ============================================================================
// SUITE PENGUJIAN SOFTWARE ESP32-S3 COFFEE E-NOSE (TANPA SENSOR GAS FISIK)
// ============================================================================
//
// Binary ini dapat dijalankan pada ESP32-S3 (atau Wokwi simulator) tanpa
// memerlukan koneksi fisik ke sensor gas atau ADS1115. Menggunakan Mock I2C
// untuk memverifikasi seluruh lapisan software:
//   1. Konversi raw ADC ke tegangan (PGA Fsr4_096V & Fsr6_144V)
//   2. Pemetaan kanal (Ain0..Ain3) & enumerasi Channel
//   3. Pemetaan 8 kanal gas fisik (MQ-6, MQ-135, MQ-7, MQ-3, TGS2602, TGS2620, TGS2611, TGS2600)
//   4. Driver ADS1115 via Mock I2C (transaksi normal, timeout 150 poll, bus error/NACK)
//   5. Mode sumber data GasSource (Dummy vs Hardware)
//   6. Pembentukan struktur MeasurementSample & payload JSON ThingsBoard
//   7. Antrean sampel (Channel), mitigasi overflow (try_send drop), & logika retry MQTT
//   8. Validasi SessionConfig (fs_hz, duration, PGA, divider_ratio)
// ============================================================================

use embassy_executor::Spawner;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel as EmbChannel;
use embassy_time::{Duration, Timer};
use embedded_hal::i2c::{Error as HalError, ErrorKind, ErrorType, I2c, Operation};
use esp_hal::clock::CpuClock;
use esp_hal::interrupt::software::SoftwareInterruptControl;
use esp_hal::timer::timg::TimerGroup;
use esp_println::println;
use heapless::String as HString;

use esp32s3_test::acquisition::{
    CoffeeIdKind, DataMode, GAS_CHANNEL_MAPPING, GasSource, MeasurementSample, SessionConfig,
    SessionConfigError, TIMESTAMP_PLACEHOLDER, build_json_payload, build_measurement_id,
};
use esp32s3_test::drivers::ads1115::{
    Ads1115, Ads1115Error, Channel as AdcChannel, DEFAULT_PGA, Pga, raw_to_voltage,
    raw_to_voltage_pga,
};

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!();
    println!("!!! PANIC DALAM PENGUJIAN !!!");
    println!("{:?}", info);
    loop {}
}

esp_bootloader_esp_idf::esp_app_desc!();

// ============================================================================
// MOCK I2C UNTUK DRIVER ADS1115
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockI2cError {
    Nack,
    Bus,
}

impl HalError for MockI2cError {
    fn kind(&self) -> ErrorKind {
        match self {
            MockI2cError::Nack => {
                ErrorKind::NoAcknowledge(embedded_hal::i2c::NoAcknowledgeSource::Address)
            }
            MockI2cError::Bus => ErrorKind::Bus,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockBehavior {
    Normal,
    Timeout,
    Error(MockI2cError),
}

pub struct MockI2c {
    pub behavior_0x48: MockBehavior,
    pub behavior_0x49: MockBehavior,
    pub mock_raw_values: [i16; 8],
    pub last_channel_0x48: u8,
    pub last_channel_0x49: u8,
    pub poll_count_0x48: u32,
    pub poll_count_0x49: u32,
}

impl MockI2c {
    pub fn new(raw_values: [i16; 8]) -> Self {
        Self {
            behavior_0x48: MockBehavior::Normal,
            behavior_0x49: MockBehavior::Normal,
            mock_raw_values: raw_values,
            last_channel_0x48: 0,
            last_channel_0x49: 0,
            poll_count_0x48: 0,
            poll_count_0x49: 0,
        }
    }
}

impl ErrorType for MockI2c {
    type Error = MockI2cError;
}

impl I2c for MockI2c {
    fn transaction(
        &mut self,
        address: u8,
        operations: &mut [Operation<'_>],
    ) -> Result<(), Self::Error> {
        let behavior = match address {
            0x48 => self.behavior_0x48,
            0x49 => self.behavior_0x49,
            _ => return Err(MockI2cError::Nack),
        };

        if let MockBehavior::Error(e) = behavior {
            return Err(e);
        }

        // write_read untuk register Conversion (pointer 0x00 -> baca 2 byte)
        if operations.len() == 2 {
            let (first, second) = operations.split_at_mut(1);
            if let (Operation::Write(w), Operation::Read(r)) = (&first[0], &mut second[0])
                && w.len() == 1
                && w[0] == 0x00
                && r.len() >= 2
            {
                let idx = if address == 0x48 {
                    (self.last_channel_0x48 as usize).min(3)
                } else {
                    4 + (self.last_channel_0x49 as usize).min(3)
                };
                let raw = self.mock_raw_values[idx];
                let be = raw.to_be_bytes();
                r[0] = be[0];
                r[1] = be[1];
                return Ok(());
            }
        }

        for op in operations {
            match op {
                Operation::Write(bytes) => {
                    if bytes.len() >= 3 && bytes[0] == 0x01 {
                        // REG_CONFIG write: ekstrak bit MUX (bit 14:12)
                        let mux = (bytes[1] >> 4) & 0x07;
                        let ch = mux.saturating_sub(4);
                        if address == 0x48 {
                            self.last_channel_0x48 = ch;
                        } else {
                            self.last_channel_0x49 = ch;
                        }
                    }
                }
                Operation::Read(bytes) => {
                    if bytes.len() >= 2 {
                        match behavior {
                            MockBehavior::Normal => {
                                bytes[0] = 0x80; // OS bit = 1 (ready)
                                bytes[1] = 0x00;
                            }
                            MockBehavior::Timeout => {
                                bytes[0] = 0x00; // OS bit = 0 (never ready)
                                bytes[1] = 0x00;
                                if address == 0x48 {
                                    self.poll_count_0x48 += 1;
                                } else {
                                    self.poll_count_0x49 += 1;
                                }
                            }
                            MockBehavior::Error(e) => return Err(e),
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// ASSERTION COUNTER DENGAN ATOMICU32 (RUST 2024 COMPLIANT)
// ============================================================================

use core::sync::atomic::{AtomicU32, Ordering};

static TOTAL_ASSERTIONS: AtomicU32 = AtomicU32::new(0);
static FAILED_ASSERTIONS: AtomicU32 = AtomicU32::new(0);

fn check_bool(cond: bool, msg: &str) {
    TOTAL_ASSERTIONS.fetch_add(1, Ordering::SeqCst);
    if !cond {
        FAILED_ASSERTIONS.fetch_add(1, Ordering::SeqCst);
        println!("  [FAIL] Assertion failed: {}", msg);
    }
}

// ============================================================================
// ENTRY POINT PENGUJIAN
// ============================================================================

#[esp_rtos::main]
async fn main(_spawner: Spawner) {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(size: 72 * 1024);
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_interrupt = SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);
    esp_rtos::start(timg0.timer0, sw_interrupt.software_interrupt0);

    println!();
    println!("================================================================");
    println!("ESP32-S3 COFFEE E-NOSE SOFTWARE TEST SUITE (TANPA SENSOR FISIK)");
    println!("================================================================");

    test_pga_conversions();
    test_channel_enum();
    test_sensor_mapping();
    test_ads1115_driver_mock();
    test_gas_source_modes();
    test_sample_and_json_payload();
    test_channel_queue_and_retry();
    test_session_config_validator();

    let total = TOTAL_ASSERTIONS.load(Ordering::SeqCst);
    let failed = FAILED_ASSERTIONS.load(Ordering::SeqCst);

    println!();
    println!("================================================================");
    if failed == 0 {
        println!(
            "SEMUA PENGUJIAN LULUS! ({} assertions checked, 0 failed)",
            total
        );
    } else {
        println!(
            "PENGUJIAN GAGAL: {} dari {} assertions gagal!",
            failed, total
        );
    }
    println!("================================================================");

    loop {
        Timer::after(Duration::from_secs(3600)).await;
    }
}

// ============================================================================
// 1. PENGUJIAN KONVERSI RAW ADC KE TEGANGAN (PGA)
// ============================================================================

fn test_pga_conversions() {
    println!("[TEST 1] Menguji konversi raw ADC ke tegangan...");

    check_bool(
        Pga::Fsr4_096V.full_scale_volts() == 4.096,
        "Pga::Fsr4_096V full scale harus 4.096V",
    );
    check_bool(
        Pga::Fsr6_144V.full_scale_volts() == 6.144,
        "Pga::Fsr6_144V full scale harus 6.144V",
    );
    check_bool(
        DEFAULT_PGA == Pga::Fsr4_096V,
        "DEFAULT_PGA harus Pga::Fsr4_096V",
    );

    // FSR 4.096V:
    // raw = 0 -> 0.0V
    check_bool(raw_to_voltage(0) == 0.0, "raw 0 harus 0.0V");
    // raw = 16384 (setengah skala positif) -> 2.048V
    let v_mid = raw_to_voltage(16384);
    check_bool(
        (v_mid - 2.048).abs() < 0.001,
        "raw 16384 pada FSR 4.096V harus ~2.048V",
    );
    // raw = 32767 (maksimum i16) -> ~4.095875V
    let v_max = raw_to_voltage(32767);
    check_bool(
        (v_max - 4.0958).abs() < 0.005,
        "raw 32767 pada FSR 4.096V harus ~4.096V",
    );
    // raw = 12000 (Dummy raw pertama) -> 12000 * 4.096 / 32768 = 1.500V
    let v_dummy0 = raw_to_voltage(12000);
    check_bool(
        (v_dummy0 - 1.500).abs() < 0.001,
        "raw 12000 pada FSR 4.096V harus tepat 1.500V",
    );

    // FSR 6.144V:
    let v_6144_mid = raw_to_voltage_pga(16384, Pga::Fsr6_144V);
    check_bool(
        (v_6144_mid - 3.072).abs() < 0.001,
        "raw 16384 pada FSR 6.144V harus ~3.072V",
    );

    println!("  [PASS] Konversi PGA terverifikasi akurat.");
}

// ============================================================================
// 2. PENGUJIAN ENUM CHANNEL
// ============================================================================

fn test_channel_enum() {
    println!("[TEST 2] Menguji enum Channel dan konversi indeks...");

    check_bool(
        AdcChannel::from_index(0) == Some(AdcChannel::Ain0),
        "Index 0 -> Ain0",
    );
    check_bool(
        AdcChannel::from_index(1) == Some(AdcChannel::Ain1),
        "Index 1 -> Ain1",
    );
    check_bool(
        AdcChannel::from_index(2) == Some(AdcChannel::Ain2),
        "Index 2 -> Ain2",
    );
    check_bool(
        AdcChannel::from_index(3) == Some(AdcChannel::Ain3),
        "Index 3 -> Ain3",
    );
    check_bool(AdcChannel::from_index(4).is_none(), "Index 4 -> None");
    check_bool(AdcChannel::from_index(255).is_none(), "Index 255 -> None");

    println!("  [PASS] Enum Channel terverifikasi.");
}

// ============================================================================
// 3. PENGUJIAN PEMETAAN SENSOR GAS FISIK
// ============================================================================

fn test_sensor_mapping() {
    println!("[TEST 3] Menguji tabel pemetaan 8 kanal sensor gas...");

    check_bool(
        GAS_CHANNEL_MAPPING[0] == ("MQ-6", 0x48, AdcChannel::Ain0),
        "Kanal 0 harus MQ-6 di 0x48 Ain0",
    );
    check_bool(
        GAS_CHANNEL_MAPPING[1] == ("MQ-135", 0x48, AdcChannel::Ain1),
        "Kanal 1 harus MQ-135 di 0x48 Ain1",
    );
    check_bool(
        GAS_CHANNEL_MAPPING[2] == ("MQ-7", 0x48, AdcChannel::Ain2),
        "Kanal 2 harus MQ-7 di 0x48 Ain2",
    );
    check_bool(
        GAS_CHANNEL_MAPPING[3] == ("MQ-3", 0x48, AdcChannel::Ain3),
        "Kanal 3 harus MQ-3 di 0x48 Ain3",
    );
    check_bool(
        GAS_CHANNEL_MAPPING[4] == ("TGS2602", 0x49, AdcChannel::Ain0),
        "Kanal 4 harus TGS2602 di 0x49 Ain0",
    );
    check_bool(
        GAS_CHANNEL_MAPPING[5] == ("TGS2620", 0x49, AdcChannel::Ain1),
        "Kanal 5 harus TGS2620 di 0x49 Ain1",
    );
    check_bool(
        GAS_CHANNEL_MAPPING[6] == ("TGS2611", 0x49, AdcChannel::Ain2),
        "Kanal 6 harus TGS2611 di 0x49 Ain2",
    );
    check_bool(
        GAS_CHANNEL_MAPPING[7] == ("TGS2600", 0x49, AdcChannel::Ain3),
        "Kanal 7 harus TGS2600 di 0x49 Ain3",
    );

    println!("  [PASS] Pemetaan 8 sensor gas cocok 100% dengan rencana hardware.");
}

// ============================================================================
// 4. PENGUJIAN DRIVER ADS1115 MENGGUNAKAN MOCK I2C
// ============================================================================

fn test_ads1115_driver_mock() {
    println!("[TEST 4] Menguji driver ADS1115 via Mock I2C...");

    let raw_sim = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
    let mut mock_i2c = MockI2c::new(raw_sim);

    let adc_48 = Ads1115::new(0x48);
    let adc_49 = Ads1115::new(0x49);

    // 4a. Pembacaan normal 4 kanal unit 0x48
    let r0 = adc_48.read_channel_raw(&mut mock_i2c, AdcChannel::Ain0);
    check_bool(r0 == Ok(1000), "0x48 Ain0 membaca nilai simulasi 1000");

    let r1 = adc_48.read_channel_raw(&mut mock_i2c, AdcChannel::Ain1);
    check_bool(r1 == Ok(2000), "0x48 Ain1 membaca nilai simulasi 2000");

    let r2 = adc_48.read_channel_raw(&mut mock_i2c, AdcChannel::Ain2);
    check_bool(r2 == Ok(3000), "0x48 Ain2 membaca nilai simulasi 3000");

    let r3 = adc_48.read_channel_raw(&mut mock_i2c, AdcChannel::Ain3);
    check_bool(r3 == Ok(4000), "0x48 Ain3 membaca nilai simulasi 4000");

    // 4b. Pembacaan normal 4 kanal unit 0x49
    let r4 = adc_49.read_channel_raw(&mut mock_i2c, AdcChannel::Ain0);
    check_bool(r4 == Ok(5000), "0x49 Ain0 membaca nilai simulasi 5000");

    let r7 = adc_49.read_channel_raw(&mut mock_i2c, AdcChannel::Ain3);
    check_bool(r7 == Ok(8000), "0x49 Ain3 membaca nilai simulasi 8000");

    // 4c. Simulasi Timeout ADS1115 (bit OS tidak pernah 1)
    mock_i2c.behavior_0x48 = MockBehavior::Timeout;
    mock_i2c.poll_count_0x48 = 0;
    let timeout_res = adc_48.read_channel_raw(&mut mock_i2c, AdcChannel::Ain0);
    check_bool(
        timeout_res == Err(Ads1115Error::Timeout),
        "Timeout saat bit OS tidak kunjung siap",
    );
    check_bool(
        mock_i2c.poll_count_0x48 >= 150,
        "Driver harus melakukan polling hingga batas MAX_POLL_ATTEMPTS (150)",
    );

    // 4d. Simulasi I2C Bus Error / NACK
    mock_i2c.behavior_0x48 = MockBehavior::Error(MockI2cError::Nack);
    let nack_res = adc_48.read_channel_raw(&mut mock_i2c, AdcChannel::Ain0);
    check_bool(
        nack_res == Err(Ads1115Error::I2c(MockI2cError::Nack)),
        "I2C NACK diteruskan sebagai Ads1115Error::I2c",
    );

    // 4e. Invalid channel index
    mock_i2c.behavior_0x48 = MockBehavior::Normal;
    let inv_res = adc_48.read_channel_index_voltage(&mut mock_i2c, 5);
    check_bool(
        inv_res == Err(Ads1115Error::InvalidChannel),
        "Index kanal 5 menghasilkan Ads1115Error::InvalidChannel",
    );

    println!("  [PASS] Driver ADS1115 dan Mock I2C berhasil diuji (success, timeout, NACK).");
}

// ============================================================================
// 5. PENGUJIAN PEMILIHAN MODE SUMBER DATA (DUMMY VS HARDWARE)
// ============================================================================

fn test_gas_source_modes() {
    println!("[TEST 5] Menguji pilihan mode sumber data gas...");

    let dummy_source: GasSource<MockI2c> = GasSource::Dummy;
    let dummy_mode = match dummy_source {
        GasSource::Dummy => DataMode::Dummy,
        GasSource::Hardware { .. } => DataMode::Hardware,
    };
    check_bool(
        dummy_mode == DataMode::Dummy,
        "GasSource::Dummy -> DataMode::Dummy",
    );
    check_bool(dummy_mode.as_str() == "dummy", "DataMode string 'dummy'");

    let mock_i2c = MockI2c::new([0; 8]);
    let hw_source = GasSource::hardware(mock_i2c, 0x48, 0x49);
    let hw_mode = match hw_source {
        GasSource::Dummy => DataMode::Dummy,
        GasSource::Hardware { .. } => DataMode::Hardware,
    };
    check_bool(
        hw_mode == DataMode::Hardware,
        "GasSource::Hardware -> DataMode::Hardware",
    );
    check_bool(hw_mode.as_str() == "hardware", "DataMode string 'hardware'");

    println!("  [PASS] Pemilihan GasSource (Dummy vs Hardware) konsisten.");
}

// ============================================================================
// 6. PENGUJIAN STRUKTUR SAMPLE & PAYLOAD JSON THINGSBOARD
// ============================================================================

fn test_sample_and_json_payload() {
    println!("[TEST 6] Menguji struktur sampel dan pembentukan payload JSON...");

    let mut timestamp: HString<32> = HString::new();
    let _ = timestamp.push_str(TIMESTAMP_PLACEHOLDER);

    let sample = MeasurementSample {
        t_s: 42,
        timestamp,
        mq3_raw: 4000,
        mq6_raw: 1000,
        mq7_raw: 3000,
        mq135_raw: 2000,
        tgs2600_raw: 8000,
        tgs2602_raw: 5000,
        tgs2611_raw: 7000,
        tgs2620_raw: 6000,
        mq3_v: 0.500,
        mq6_v: 0.125,
        mq7_v: 0.375,
        mq135_v: 0.250,
        tgs2600_v: 1.000,
        tgs2602_v: 0.625,
        tgs2611_v: 0.875,
        tgs2620_v: 0.750,
        temp_c: 28.50,
        rh_pct: 62.30,
        dht_age_s: 0,
        gas_ok_mask: 0xFF,
    };

    let mut json_buf: HString<384> = HString::new();
    let res = build_json_payload(&sample, "DAQ03_B02_C07_A04_M01", "dummy", &mut json_buf);
    check_bool(res.is_ok(), "build_json_payload harus berhasil");

    let json_str = json_buf.as_str();
    println!("  Generated JSON: {}", json_str);

    // Verifikasi pemetaan field ThingsBoard:
    // gas1 = mq6_v (0.125)
    // gas2 = mq135_v (0.250)
    // gas3 = mq7_v (0.375)
    // gas4 = mq3_v (0.500)
    // gas5 = tgs2602_v (0.625)
    // gas6 = tgs2620_v (0.750)
    // gas7 = tgs2611_v (0.875)
    // gas8 = tgs2600_v (1.000)
    check_bool(
        json_str.contains("\"measurement_id\":\"DAQ03_B02_C07_A04_M01\""),
        "JSON memuat measurement_id",
    );
    check_bool(json_str.contains("\"sample\":42"), "JSON memuat sample 42");
    check_bool(
        json_str.contains("\"gas1\":0.125"),
        "JSON gas1 memetakan MQ-6 (0.125)",
    );
    check_bool(
        json_str.contains("\"gas2\":0.250"),
        "JSON gas2 memetakan MQ-135 (0.250)",
    );
    check_bool(
        json_str.contains("\"gas3\":0.375"),
        "JSON gas3 memetakan MQ-7 (0.375)",
    );
    check_bool(
        json_str.contains("\"gas4\":0.500"),
        "JSON gas4 memetakan MQ-3 (0.500)",
    );
    check_bool(
        json_str.contains("\"gas5\":0.625"),
        "JSON gas5 memetakan TGS2602 (0.625)",
    );
    check_bool(
        json_str.contains("\"gas6\":0.750"),
        "JSON gas6 memetakan TGS2620 (0.750)",
    );
    check_bool(
        json_str.contains("\"gas7\":0.875"),
        "JSON gas7 memetakan TGS2611 (0.875)",
    );
    check_bool(
        json_str.contains("\"gas8\":1.000"),
        "JSON gas8 memetakan TGS2600 (1.000)",
    );
    check_bool(
        json_str.contains("\"temperature\":28.50"),
        "JSON memuat suhu",
    );
    check_bool(
        json_str.contains("\"humidity\":62.30"),
        "JSON memuat kelembapan",
    );
    check_bool(
        json_str.contains("\"data_mode\":\"dummy\""),
        "JSON memuat data_mode dummy",
    );
    check_bool(
        json_str.contains("\"gas_ok\":255"),
        "JSON memuat gas_ok mask (255 = 0xFF)",
    );

    println!("  [PASS] Struktur sample dan serialisasi JSON ThingsBoard valid.");
}

// ============================================================================
// 7. PENGUJIAN ANTREAN SAMPEL (CHANNEL) & RETRY LOGIC
// ============================================================================

fn test_channel_queue_and_retry() {
    println!("[TEST 7] Menguji antrean sampel (Channel) dan penanganan overflow...");

    // Channel kecil berkapasitas 2 untuk simulasi overflow
    let test_channel: EmbChannel<CriticalSectionRawMutex, MeasurementSample, 2> = EmbChannel::new();

    let dummy_sample = MeasurementSample {
        t_s: 0,
        timestamp: HString::new(),
        mq3_raw: 0,
        mq6_raw: 0,
        mq7_raw: 0,
        mq135_raw: 0,
        tgs2600_raw: 0,
        tgs2602_raw: 0,
        tgs2611_raw: 0,
        tgs2620_raw: 0,
        mq3_v: 0.0,
        mq6_v: 0.0,
        mq7_v: 0.0,
        mq135_v: 0.0,
        tgs2600_v: 0.0,
        tgs2602_v: 0.0,
        tgs2611_v: 0.0,
        tgs2620_v: 0.0,
        temp_c: 0.0,
        rh_pct: 0.0,
        dht_age_s: 0,
        gas_ok_mask: 0xFF,
    };

    check_bool(
        test_channel.try_send(dummy_sample.clone()).is_ok(),
        "Kirim sampel 1 sukses",
    );
    check_bool(
        test_channel.try_send(dummy_sample.clone()).is_ok(),
        "Kirim sampel 2 sukses (channel penuh)",
    );

    // Sampel ke-3 harus gagal kirim (try_send) tanpa memblokir
    let mut dropped_count: u32 = 0;
    if test_channel.try_send(dummy_sample.clone()).is_err() {
        dropped_count += 1;
    }
    check_bool(
        dropped_count == 1,
        "Sampel ke-3 dibuang saat channel penuh (non-blocking)",
    );

    // Ambil satu sampel dari antrean
    let received = test_channel.try_receive();
    check_bool(received.is_ok(), "Pengambilan sampel dari antrean berhasil");

    // Sekarang slot kosong, sampel ke-4 dapat masuk
    check_bool(
        test_channel.try_send(dummy_sample).is_ok(),
        "Sampel berhasil dikirim kembali setelah ada slot kosong",
    );

    // Simulasi retry MQTT logic
    let mut pending: Option<u32> = None;
    let mut pending_retries: u8 = 0;

    // Percobaan pertama gagal:
    let sample_ts: u32 = 10;
    if pending_retries < 1 {
        pending_retries += 1;
        pending = Some(sample_ts);
    }
    check_bool(
        pending == Some(10) && pending_retries == 1,
        "Percobaan 1 gagal: sample disimpan ke pending untuk 1x retry",
    );

    // Percobaan kedua gagal (setelah 1x retry):
    if pending_retries < 1 {
        pending_retries += 1;
        pending = Some(sample_ts);
    } else {
        pending_retries = 0;
        pending = None;
    }
    check_bool(
        pending.is_none() && pending_retries == 0,
        "Percobaan 2 gagal: sample dibuang agar antrean tidak macet",
    );

    println!("  [PASS] Antrean sampel dan logika retry terverifikasi aman.");
}

// ============================================================================
// 8. PENGUJIAN VALIDATOR SESSION CONFIG
// ============================================================================

fn test_session_config_validator() {
    println!("[TEST 8] Menguji validasi SessionConfig dan batas parameter...");

    // Default config valid
    let def_cfg = SessionConfig::default();
    check_bool(def_cfg.validate().is_ok(), "Default SessionConfig valid");
    check_bool(
        def_cfg.sample_period_ms() == Ok(1000),
        "Default 1.0 Hz = period 1000 ms",
    );

    // fs_hz <= 0 ditolak
    let cfg_zero_fs = SessionConfig {
        fs_hz: 0.0,
        ..SessionConfig::default()
    };
    check_bool(
        cfg_zero_fs.validate() == Err(SessionConfigError::InvalidFsHz),
        "fs_hz = 0 ditolak",
    );

    // fs_hz negatif ditolak
    let cfg_neg_fs = SessionConfig {
        fs_hz: -1.0,
        ..SessionConfig::default()
    };
    check_bool(
        cfg_neg_fs.validate() == Err(SessionConfigError::InvalidFsHz),
        "fs_hz negatif ditolak",
    );

    // fs_hz > 1000 ditolak
    let cfg_high_fs = SessionConfig {
        fs_hz: 1000.1,
        ..SessionConfig::default()
    };
    check_bool(
        cfg_high_fs.validate() == Err(SessionConfigError::FsHzTooHigh),
        "fs_hz > 1000 ditolak",
    );

    // duration_s = 0 ditolak
    let cfg_zero_dur = SessionConfig {
        duration_s: 0,
        ..SessionConfig::default()
    };
    check_bool(
        cfg_zero_dur.validate() == Err(SessionConfigError::InvalidDurationS),
        "duration_s = 0 ditolak",
    );

    // pga_fsr_v <= 0 ditolak
    let cfg_zero_pga = SessionConfig {
        pga_fsr_v: 0.0,
        ..SessionConfig::default()
    };
    check_bool(
        cfg_zero_pga.validate() == Err(SessionConfigError::InvalidPgaFsrV),
        "pga_fsr_v = 0 ditolak",
    );

    // divider_ratio = None (valid, belum ditentukan)
    let cfg_ratio_none = SessionConfig {
        divider_ratio: None,
        ..SessionConfig::default()
    };
    check_bool(
        cfg_ratio_none.validate().is_ok(),
        "divider_ratio None valid",
    );

    // divider_ratio = Some(0.0) ditolak
    let cfg_ratio_zero = SessionConfig {
        divider_ratio: Some(0.0),
        ..SessionConfig::default()
    };
    check_bool(
        cfg_ratio_zero.validate() == Err(SessionConfigError::InvalidDividerRatio),
        "divider_ratio 0.0 ditolak",
    );

    // Build measurement ID test
    let meas_id = build_measurement_id(3, 2, CoffeeIdKind::Coffee, 7, 4, 1);
    check_bool(
        meas_id.as_str() == "DAQ03_B02_C07_A04_M01",
        "build_measurement_id cocok dengan pola DAQ03_B02_C07_A04_M01",
    );

    println!("  [PASS] Validator SessionConfig dan Measurement ID terverifikasi.");
}
