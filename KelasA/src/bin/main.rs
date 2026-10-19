#![no_std]
#![no_main]

// ============================================================================
// ESP32-S3 -> Wi-Fi -> DHCP -> DNS -> MQTT THINGSBOARD CLOUD
// ============================================================================
//
// Tiga task berjalan concurrent lewat Embassy executor:
//   - net_task         : network stack (Wi-Fi + DHCP), tidak pernah selesai.
//   - acquisition_task : menjalankan Acquisition::run_session() (logika ada di
//                        acquisition.rs): 300 sample, 1 Hz, deadline tetap.
//                        Gas: dummy atau ADS1115 (lihat GasSource di main()).
//                        Suhu/RH: DHT22 chamber (+ ambient sebagai cadangan).
//                        Mengirim measurement_id + data_mode SEKALI ke
//                        mqtt_task lewat SESSION_INFO sebelum sesi mulai.
//   - mqtt_task        : resolve DNS mqtt.thingsboard.cloud, connect TCP,
//                        MQTT CONNECT dengan Access Token sebagai username,
//                        tunggu SESSION_INFO sekali, lalu publish tiap
//                        Sample dari Channel ke v1/devices/me/telemetry
//                        (QoS 0), menyertakan measurement_id + data_mode.
//
// TinyML (src/tinyml): fitur disuap per detik di Acquisition::run_session();
// setelah sesi selesai acquisition_task menjalankan inference SEKALI dan
// mengirim JSON hasilnya lewat ML_RESULT. mqtt_task mem-publish JSON itu
// sebagai pesan TERPISAH (bukan field tambahan di payload per-sample).

use embassy_executor::Spawner;
use embassy_mqtt_lite::{ConnectOptions, MqttClient};
use embassy_net::dns::DnsQueryType;
use embassy_net::tcp::TcpSocket;
use embassy_net::{Config as NetConfig, IpAddress, Runner, StackResources};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Timer, with_timeout};

use esp_hal::Blocking;
use esp_hal::clock::CpuClock;
use esp_hal::gpio::Flex;
use esp_hal::i2c::master::I2c;
use esp_hal::interrupt::software::SoftwareInterruptControl;
use esp_hal::rng::Rng;
use esp_hal::timer::timg::TimerGroup;

use esp_println::println;

use esp_radio::wifi::sta::StationConfig;
use esp_radio::wifi::{self, Config as WifiConfig};

use esp32s3_test::acquisition::{
    Acquisition, CoffeeIdKind, DataMode, GasSource, MeasurementIdString, MeasurementMetadata,
    MeasurementSample, SessionConfig, build_json_payload, build_measurement_id,
};

use esp32s3_test::tinyml::inference::classify_session;
use esp32s3_test::tinyml::model_params::MODEL;
use esp32s3_test::tinyml::telemetry::write_ml_json;

use esp32s3_test::drivers::dht22::Dht22;

use heapless::String as HString;
use static_cell::StaticCell;

// ============================================================================
// KONFIGURASI
// ============================================================================

use esp32s3_test::config::{TB_ACCESS_TOKEN, WIFI_PASSWORD, WIFI_SSID};

// ThingsBoard Cloud. Hostname di-resolve via DNS setiap kali (re)connect.
const TB_HOST: &str = "mqtt.thingsboard.cloud";
const TB_PORT: u16 = 1883;
const TB_TOPIC: &str = "v1/devices/me/telemetry";
const MQTT_CLIENT_ID: &str = "esp32s3-coffee-enose";

/// Tipe bus I2C untuk ADS1115 (dipakai walau mode gas masih Dummy).
type GasI2c = I2c<'static, Blocking>;

// ============================================================================
// CHANNEL + SESSION INFO
// ============================================================================

// acquisition_task (producer) -> mqtt_task (consumer). Satu entri per sample.
// Kapasitas ditingkatkan ke 128 sampel (toleransi buffering >2 menit gangguan jaringan)
// dengan alokasi RAM hanya ~12 kB.
static SAMPLE_CHANNEL: Channel<CriticalSectionRawMutex, MeasurementSample, 128> = Channel::new();

/// Identitas sesi (measurement_id + data_mode), dikirim SEKALI oleh
/// acquisition_task di awal sesi (sebelum run_session dipanggil), dibaca
/// SEKALI oleh mqtt_task sebelum masuk publish loop. `Signal` (bukan
/// `Channel`) dipakai karena nilainya konstan sepanjang sesi -- bukan
/// aliran data per-sample.
struct SessionInfo {
    measurement_id: MeasurementIdString,
    data_mode: &'static str,
}

static SESSION_INFO: Signal<CriticalSectionRawMutex, SessionInfo> = Signal::new();

/// JSON hasil TinyML (satu per sesi), dibangun acquisition_task setelah sesi
/// selesai, dipublish mqtt_task. Maks 256 byte (batas buffer, bukan batas paket).
type MlJson = HString<256>;
static ML_RESULT: Signal<CriticalSectionRawMutex, MlJson> = Signal::new();

// ============================================================================
// PANIC HANDLER
// ============================================================================

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

esp_bootloader_esp_idf::esp_app_desc!();

// ============================================================================
// MAIN
// ============================================================================

#[esp_rtos::main]
async fn main(spawner: Spawner) {
    // 1. ESP HAL INIT
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    // ------------------------------------------------------------------------
    // DHT22 #1 chamber = GPIO10 (dataset utama)
    // Catatan: DHT22 ambient (GPIO11) belum tersedia secara fisik,
    // sehingga jangan diinisialisasi atau dianggap terpasang.
    // ------------------------------------------------------------------------
    let dht22_chamber = Dht22::new(Flex::new(peripherals.GPIO10));

    // ------------------------------------------------------------------------
    // SUMBER GAS (PILIHAN EKSPLISIT: DUMMY ATAU HARDWARE ADS1115)
    // ------------------------------------------------------------------------
    // Konfigurasi pin I2C terencana (mudah diubah jika pin final berbeda):
    //   SDA: GPIO8, SCL: GPIO9
    // Alamat ADS1115 (Unit Kanan #1 @ 0x48, Unit Kiri #2 @ 0x49):
    //   First (0x48):  MQ-6 (A0),   MQ-135 (A1), MQ-7 (A2),    MQ-3 (A3)
    //   Second (0x49): TGS2602 (A0), TGS2620 (A1), TGS2611 (A2), TGS2600 (A3)
    // Mode dummy (USE_HARDWARE_GAS = false): TIDAK ada transaksi I2C.
    // Mode hardware (USE_HARDWARE_GAS = true): inisialisasi I2C0 dan baca ADC fisik.
    const USE_HARDWARE_GAS: bool = false;
    const ADS1115_ADDR_FIRST: u8 = 0x48;
    const ADS1115_ADDR_SECOND: u8 = 0x49;
    const I2C_FREQ_KHZ: u32 = 100;

    let gas_source: GasSource<GasI2c> = if USE_HARDWARE_GAS {
        let i2c = I2c::new(
            peripherals.I2C0,
            esp_hal::i2c::master::Config::default()
                .with_frequency(esp_hal::time::Rate::from_khz(I2C_FREQ_KHZ)),
        )
        .expect("Gagal inisialisasi I2C0")
        .with_sda(peripherals.GPIO8)
        .with_scl(peripherals.GPIO9);
        GasSource::hardware(i2c, ADS1115_ADDR_FIRST, ADS1115_ADDR_SECOND)
    } else {
        GasSource::Dummy
    };

    let acquisition = Acquisition::new(gas_source, Some(dht22_chamber));

    // 2. HEAP ALLOCATOR
    esp_alloc::heap_allocator!(size: 72 * 1024);

    // 3. TIMER GROUP + 4. SOFTWARE INTERRUPT CONTROL
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_interrupt = SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);

    // 5. ESP-RTOS START
    esp_rtos::start(timg0.timer0, sw_interrupt.software_interrupt0);

    // 6. WIFI NEW
    let (mut controller, interfaces) =
        wifi::new(peripherals.WIFI, Default::default()).expect("Gagal inisialisasi radio Wi-Fi");

    println!("Wi-Fi controller initialized");

    // 7. STATION CONFIG (set_config() sendiri yang men-(re)start controller)
    let sta_config = WifiConfig::Station(
        StationConfig::default()
            .with_ssid(WIFI_SSID)
            .with_password(WIFI_PASSWORD.into()),
    );

    controller
        .set_config(&sta_config)
        .expect("Gagal set konfigurasi Wi-Fi (set_config)");

    // Seed acak untuk stack IP (bukan untuk keamanan kriptografis).
    let rng = Rng::new();
    let net_seed = ((rng.random() as u64) << 32) | rng.random() as u64;

    // 8. EMBASSY-NET STACK (3 slot: DHCP + DNS + TCP)
    let net_config = NetConfig::dhcpv4(Default::default());

    static RESOURCES: StaticCell<StackResources<3>> = StaticCell::new();

    let (stack, runner) = embassy_net::new(
        interfaces.station,
        net_config,
        RESOURCES.init(StackResources::new()),
        net_seed,
    );

    // 9. NET TASK
    spawner.spawn(net_task(runner).expect("Gagal membuat net_task"));

    // 10. CONNECT WIFI (satu-satunya connect_async())
    println!("Menghubungkan ke access point \"{}\"...", WIFI_SSID);
    println!("Sebelum connect_async()");

    controller
        .connect_async()
        .await
        .expect("Gagal konek ke access point - cek SSID/password");

    println!("Sesudah connect_async()");
    println!("Wi-Fi connected");

    // 11. DHCP
    println!("Menunggu DHCP...");
    stack.wait_config_up().await;

    // 12. IP
    let ip_config = stack.config_v4().expect("config_v4 kosong setelah DHCP");

    println!("IP address: {}", ip_config.address.address());
    println!("DHCP OK");

    // 13. MQTT TASK + 14. ACQUISITION TASK
    spawner.spawn(mqtt_task(stack).expect("Gagal membuat mqtt_task"));

    spawner.spawn(acquisition_task(acquisition).expect("Gagal membuat acquisition_task"));

    // 15. IDLE
    loop {
        Timer::after(Duration::from_secs(3600)).await;
    }
}

// ============================================================================
// NETWORK TASK
// ============================================================================

#[embassy_executor::task]
async fn net_task(mut runner: Runner<'static, esp_radio::wifi::Interface<'static>>) -> ! {
    runner.run().await;
}

// ============================================================================
// ACQUISITION TASK
// ============================================================================

/// Seluruh logika (timing deadline 1 Hz, 300 sample, pembacaan sensor,
/// penanganan error) ada di `Acquisition::run_session()` (acquisition.rs).
#[embassy_executor::task]
async fn acquisition_task(mut acquisition: Acquisition<GasI2c>) {
    let config = SessionConfig {
        heating_s: 0,
        duration_s: 300,
        fs_hz: 1.0,
        ..SessionConfig::default()
    };

    // Validasi SEBELUM membangun metadata/mengirim SESSION_INFO. Dipakai
    // sample_period_ms() (BUKAN validate() saja) karena sample_period_ms()
    // memanggil validate() lebih dulu LALU mengecek PeriodOverflow --
    // kasus yang lolos validate() tapi tetap gagal dihitung periodenya
    // (fs_hz valid tapi ekstrem kecil). Kalau dicek pakai validate() saja,
    // celah itu baru ketahuan di dalam run_session(), padahal SESSION_INFO
    // sudah kadung dikirim ke mqtt_task.
    if let Err(e) = config.sample_period_ms() {
        println!();
        println!("=== SESSION TIDAK DIMULAI ===");
        println!("SessionConfig tidak valid: {:?}", e);
        println!("Acquisition idle -- perbaiki SessionConfig lalu reboot/reflash.");
        loop {
            Timer::after(Duration::from_secs(3600)).await;
        }
    }

    // Identitas sesi contoh. Ganti sesuai eksperimen yang sebenarnya.
    // heating_s/duration_s/fs_hz/pga_fsr_v/divider_ratio diambil LANGSUNG
    // dari `config` (bukan diketik ulang) supaya metadata tidak pernah
    // bisa berbeda dari config yang benar-benar dipakai run_session().
    let metadata = MeasurementMetadata {
        measurement_id: build_measurement_id(3, 2, CoffeeIdKind::Coffee, 7, 4, 1),
        daq_id: 3,
        batch_id: 2,
        group_id: HString::<16>::try_from("G01").unwrap(),
        coffee_id: HString::<16>::try_from("C07").unwrap(),
        species: HString::<32>::try_from("UNKNOWN").unwrap(),
        bean_state: HString::<32>::try_from("UNKNOWN").unwrap(),
        aliquot_id: 4,
        mass_g: None,
        timestamp: HString::<32>::try_from("PENDING_HOST_TS").unwrap(),
        temperature: None,
        humidity: None,
        room_temp_room_rh: HString::<32>::try_from("UNKNOWN").unwrap(),
        // Pemanasan FISIK (di luar firmware), dalam menit, sesuai arahan
        // dosen -- INFORMASIONAL SAJA, dicatat manual, BUKAN diukur atau
        // dijalankan oleh firmware. TIDAK berkaitan dengan config.heating_s
        // (sekarang 0, artinya firmware tidak melakukan delay pemanasan
        // internal apa pun -- lihat juga dokumentasi SessionConfig).
        warmup_min: 5.0,
        interval_min: 0.0,
        run_order: 1,
        heating_s: config.heating_s,
        duration_s: config.duration_s,
        fs_hz: config.fs_hz,
        pga_fsr_v: config.pga_fsr_v,
        divider_ratio: config.divider_ratio,
        protocol_ver: HString::<16>::try_from("v1").unwrap(),
        operator: HString::<32>::try_from("UNKNOWN").unwrap(),
        qc_flag: HString::<16>::try_from("PENDING").unwrap(),
        notes: HString::<64>::new(),
    };

    // Kirim identitas sesi ke mqtt_task SEKALI, sebelum sesi mulai.
    SESSION_INFO.signal(SessionInfo {
        measurement_id: metadata.measurement_id.clone(),
        data_mode: acquisition.data_mode().as_str(),
    });

    match acquisition
        .run_session(&metadata, config, &SAMPLE_CHANNEL)
        .await
    {
        Ok(()) => {
            println!();
            println!("=== SESSION COMPLETE ===");
            println!(
                "Measurement_ID: {} -- {} sample terkirim ke channel.",
                metadata.measurement_id, config.duration_s
            );

            // ---- TinyML: inference SEKALI setelah sesi selesai (tidak mengganggu
            // timing sampling karena sampling sudah berhenti).
            let data_mode = acquisition.data_mode();
            let outcome = classify_session(
                acquisition.session_features(),
                &MODEL,
                data_mode == DataMode::Dummy,
            );
            match outcome.prediction {
                Some(p) => println!(
                    "ML: status={} class={} score={:.4}",
                    outcome.status.as_str(),
                    p.label,
                    p.score
                ),
                None => println!(
                    "ML: status={} (TIDAK ada prediksi)",
                    outcome.status.as_str()
                ),
            }
            if data_mode == DataMode::Dummy {
                println!("ML: data DUMMY -- hasil hanya uji integrasi, BUKAN klasifikasi aroma.");
            }

            let mut ml_json: MlJson = HString::new();
            match write_ml_json(
                &mut ml_json,
                metadata.measurement_id.as_str(),
                data_mode.as_str(),
                &MODEL,
                &outcome,
                acquisition.session_frames(),
            ) {
                Ok(()) => ML_RESULT.signal(ml_json),
                Err(_) => println!("ML: JSON melebihi buffer -- tidak dikirim"),
            }
        }
        Err(e) => {
            // Secara praktis tidak akan tercapai: config sudah lolos
            // sample_period_ms() di atas, dan run_session() memanggil
            // fungsi murni yang sama terhadap config yang sama persis.
            // Ditangani untuk kelengkapan, bukan jalur yang diharapkan.
            println!();
            println!("=== SESSION GAGAL: konfigurasi tidak valid: {:?} ===", e);
        }
    }

    println!();
    println!("Acquisition task selesai -- TIDAK otomatis mengulang sesi.");
    println!("Alasan: satu sesi terikat ke satu Aliquot/Coffee fisik; mengulang");
    println!("otomatis akan mengukur kondisi fisik yang sama berulang kali tanpa");
    println!("operator mengganti sampel. Belum ada mekanisme trigger (mis. RPC");
    println!("MQTT) untuk memulai sesi baru tanpa reboot.");
    println!("Untuk sesi berikutnya: reboot/reflash ESP32 dengan metadata baru.");

    loop {
        Timer::after(Duration::from_secs(60)).await;
        println!("[heartbeat] Acquisition idle, menunggu reboot untuk sesi baru.");
    }
}

// ============================================================================
// MQTT TASK
// ============================================================================

#[embassy_executor::task]
async fn mqtt_task(stack: embassy_net::Stack<'static>) -> ! {
    println!();
    println!("=== MQTT TASK START ===");

    println!("Menunggu session info (measurement_id/data_mode) dari acquisition_task...");
    let session_info = SESSION_INFO.wait().await;
    println!(
        "Session info diterima: measurement_id={}, data_mode={}",
        session_info.measurement_id, session_info.data_mode
    );

    let mut json_buf: HString<384> = HString::new();
    let mut pending: Option<MeasurementSample> = None;
    let mut pending_retries: u8 = 0;
    let mut pending_ml: Option<MlJson> = None;
    let mut ml_retries: u8 = 0;

    loop {
        println!();
        println!("DNS resolving {}...", TB_HOST);

        let resolved = match stack.dns_query(TB_HOST, DnsQueryType::A).await {
            Ok(addrs) => addrs,
            Err(e) => {
                println!("DNS resolve gagal: {:?} -- coba lagi dalam 3 detik", e);
                Timer::after(Duration::from_secs(3)).await;
                continue;
            }
        };

        let broker_addr: IpAddress = match resolved.first() {
            Some(addr) => *addr,
            None => {
                println!(
                    "DNS resolve tidak mengembalikan alamat apa pun -- coba lagi dalam 3 detik"
                );
                Timer::after(Duration::from_secs(3)).await;
                continue;
            }
        };

        println!("DNS resolved");
        println!("Connecting to ThingsBoard...");

        let mut rx_buffer = [0u8; 2048];
        let mut tx_buffer = [0u8; 2048];

        let mut socket = TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(10)));

        if let Err(e) = socket.connect((broker_addr, TB_PORT)).await {
            println!("MQTT TCP connect gagal: {:?} -- coba lagi dalam 3 detik", e);
            Timer::after(Duration::from_secs(3)).await;
            continue;
        }

        println!("MQTT TCP connected");

        let mut client = MqttClient::new(&mut socket);

        let connect_options = ConnectOptions {
            username: Some(TB_ACCESS_TOKEN),
            password: None,
            last_will: None,
        };

        if let Err(e) = client
            .connect_with_options(MQTT_CLIENT_ID, 60, &connect_options)
            .await
        {
            println!("MQTT CONNECT gagal: {:?} -- coba lagi dalam 3 detik", e);
            Timer::after(Duration::from_secs(3)).await;
            continue;
        }

        println!("MQTT connected");

        loop {
            let sample = match pending.take() {
                Some(s) => s,
                None => match with_timeout(Duration::from_secs(2), SAMPLE_CHANNEL.receive()).await {
                    Ok(s) => s,
                    Err(_) => {
                        // Channel kosong >= 2 detik: kirim laporan ML (jika ada).
                        // Sengaja menunggu channel kosong supaya laporan tidak
                        // mendahului sample terakhir sesi.
                        if pending_ml.is_none() {
                            pending_ml = ML_RESULT.try_take();
                        }
                        if let Some(ml) = pending_ml.take() {
                            println!("MQTT ML PAYLOAD: {}", ml.as_str());
                            let res = client.publish(TB_TOPIC, ml.as_bytes()).await;
                            match res {
                                Ok(()) => {
                                    ml_retries = 0;
                                    println!("MQTT publish ML OK");
                                }
                                Err(e) => {
                                    println!("MQTT publish ML gagal: {:?} -- reconnect", e);
                                    if ml_retries < 1 {
                                        ml_retries += 1;
                                        pending_ml = Some(ml);
                                    } else {
                                        ml_retries = 0;
                                        println!("Laporan ML dibuang setelah 1x retry");
                                    }
                                    break;
                                }
                            }
                        }
                        continue;
                    }
                },
            };
            let t_s = sample.t_s;

            if build_json_payload(
                &sample,
                session_info.measurement_id.as_str(),
                session_info.data_mode,
                &mut json_buf,
            )
            .is_err()
            {
                println!("Gagal membangun payload JSON untuk sample={}", t_s);
                continue;
            }

            println!("MQTT PAYLOAD: {}", json_buf.as_str());

            match client.publish(TB_TOPIC, json_buf.as_bytes()).await {
                Ok(()) => {
                    pending_retries = 0;
                    println!("MQTT publish OK: sample={}", t_s);
                }
                Err(e) => {
                    println!("MQTT publish gagal (sample={}): {:?} -- reconnect", t_s, e);
                    // Coba SEKALI lagi setelah reconnect; setelah itu dibuang
                    // supaya sampel yang selalu gagal tidak diulang selamanya.
                    if pending_retries < 1 {
                        pending_retries += 1;
                        pending = Some(sample);
                    } else {
                        pending_retries = 0;
                        println!("sample={} dibuang setelah 1x retry", t_s);
                    }
                    break;
                }
            }
        }

        Timer::after(Duration::from_secs(3)).await;
    }
}
