#![no_std]
#![no_main]

// ============================================================================
// STEP 5B: ESP32-S3 -> Wi-Fi -> DHCP -> DNS -> MQTT THINGSBOARD CLOUD
// ============================================================================
//
// Tiga task berjalan concurrent lewat Embassy executor:
//   - net_task         : network stack (Wi-Fi + DHCP), tidak pernah selesai.
//   - acquisition_task : dummy_sensor_data() 500x, interval 1 detik (async,
//                        TIDAK busy-wait), kirim tiap Sample ke mqtt_task
//                        lewat Channel, lalu selesai/return.
//   - mqtt_task        : resolve DNS mqtt.thingsboard.cloud, connect TCP,
//                        MQTT CONNECT dengan Access Token sebagai username,
//                        publish tiap Sample yang diterima dari Channel ke
//                        v1/devices/me/telemetry (QoS 0).
//
// TIDAK ada di file ini: TLS, HTTP/HTTPS, ThingsBoard SDK, TinyML, OTA,
// sensor fisik. acquisition.rs TIDAK diubah. Struktur Wi-Fi (langkah 1-12
// di main()) TIDAK diubah karena sudah terbukti bekerja.

use core::fmt::Write as _;

use embassy_executor::Spawner;
use embassy_mqtt_lite::{ConnectOptions, MqttClient};
use embassy_net::dns::DnsQueryType;
use embassy_net::tcp::TcpSocket;
use embassy_net::{Config as NetConfig, IpAddress, Runner, StackResources};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_time::{Duration, Timer};

use esp_hal::clock::CpuClock;
use esp_hal::gpio::Flex;
use esp_hal::interrupt::software::SoftwareInterruptControl;
use esp_hal::rng::Rng;
use esp_hal::timer::timg::TimerGroup;

use esp_println::println;

use esp_radio::wifi::sta::StationConfig;
use esp_radio::wifi::{self, Config as WifiConfig};

use esp32s3_test::acquisition::{
    dummy_sensor_data,
    Sample,
    SensorData,
    TOTAL_SAMPLES,
};

use esp32s3_test::drivers::dht22::{Dht22, Dht22Reading};

use heapless::String as HString;
use static_cell::StaticCell;

// ============================================================================
// KONFIGURASI
// ============================================================================

// Diambil dari environment variable saat build (compile-time).
// Untuk sementara masih hard-coded seperti kode asli.
//
// TODO:
// Nanti bisa dipindahkan ke .env / konfigurasi build.
use esp32s3_test::config::{
    WIFI_SSID,
    WIFI_PASSWORD,
    TB_ACCESS_TOKEN,
};

// ThingsBoard Cloud.
// Hostname di-resolve via DNS setiap kali (re)connect.
const TB_HOST: &str = "mqtt.thingsboard.cloud";

const TB_PORT: u16 = 1883;

const TB_TOPIC: &str = "v1/devices/me/telemetry";

const MQTT_CLIENT_ID: &str = "esp32s3-coffee-enose";

// ============================================================================
// CHANNEL
// ============================================================================

// Channel:
// acquisition_task (producer) -> mqtt_task (consumer).
//
// Kapasitas 4 cukup untuk buffer beberapa sample kalau mqtt_task
// sedang sibuk publish.
static SAMPLE_CHANNEL: Channel<CriticalSectionRawMutex, Sample, 4> = Channel::new();

// ============================================================================
// PANIC HANDLER
// ============================================================================

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// ============================================================================
// ESP-IDF APP DESCRIPTION
// ============================================================================

esp_bootloader_esp_idf::esp_app_desc!();

// ============================================================================
// MAIN
// ============================================================================

#[esp_rtos::main]
async fn main(spawner: Spawner) {
    // ========================================================================
    // 1. ESP HAL INIT
    // ========================================================================

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());

    let peripherals = esp_hal::init(config);

    // ========================================================================
    // DHT22
    // ========================================================================
    //
    // DHT22 #1 — suhu & kelembaban ruang chamber
    // GPIO10
    //
    // DHT22 #2 — suhu & kelembaban ambient
    // GPIO11

    let dht1_pin = Flex::new(peripherals.GPIO10);
    let dht22_chamber = Dht22::new(dht1_pin);

    let dht2_pin = Flex::new(peripherals.GPIO11);
    let dht22_ambient = Dht22::new(dht2_pin);

    // ========================================================================
    // 2. HEAP ALLOCATOR
    // ========================================================================

    esp_alloc::heap_allocator!(size: 72 * 1024);

    // ========================================================================
    // 3. TIMER GROUP
    // 4. SOFTWARE INTERRUPT CONTROL
    // ========================================================================

    let timg0 = TimerGroup::new(peripherals.TIMG0);

    let sw_interrupt =
        SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);

    // ========================================================================
    // 5. ESP-RTOS START
    // ========================================================================

    esp_rtos::start(
        timg0.timer0,
        sw_interrupt.software_interrupt0,
    );

    // ========================================================================
    // 6. WIFI NEW
    // ========================================================================

    let (mut controller, interfaces) =
        wifi::new(peripherals.WIFI, Default::default())
            .expect("Gagal inisialisasi radio Wi-Fi");

    println!("Wi-Fi controller initialized");

    // ========================================================================
    // 7. SET STATION CONFIG
    // ========================================================================

    // set_config() sendiri yang men-(re)start controller.
    // Tidak ada start_async() terpisah di esp-radio 0.18.0.

    let sta_config = WifiConfig::Station(
        StationConfig::default()
            .with_ssid(WIFI_SSID)
            .with_password(WIFI_PASSWORD.into()),
    );

    controller
        .set_config(&sta_config)
        .expect("Gagal set konfigurasi Wi-Fi (set_config)");

    // ========================================================================
    // RANDOM SEED UNTUK NETWORK STACK
    // ========================================================================

    // Seed acak untuk stack IP.
    // Bukan untuk keamanan kriptografis.

    let rng = Rng::new();

    let net_seed =
        ((rng.random() as u64) << 32) | rng.random() as u64;

    // ========================================================================
    // 8. CREATE EMBASSY-NET STACK
    // ========================================================================

    // 3 slot:
    //   1 DHCP
    //   1 DNS
    //   1 TCP

    let net_config = NetConfig::dhcpv4(Default::default());

    static RESOURCES: StaticCell<StackResources<3>> =
        StaticCell::new();

    let (stack, runner) = embassy_net::new(
        interfaces.station,
        net_config,
        RESOURCES.init(StackResources::new()),
        net_seed,
    );

    // ========================================================================
    // 9. SPAWN NET TASK
    // ========================================================================

    // Harus tetap berjalan di background sepanjang program.

    spawner
        .spawn(net_task(runner).expect("Gagal membuat net_task"));

    // ========================================================================
    // 10. CONNECT WIFI
    // ========================================================================

    println!(
        "Menghubungkan ke access point \"{}\"...",
        WIFI_SSID
    );

    println!("Sebelum connect_async()");

    controller
        .connect_async()
        .await
        .expect("Gagal konek ke access point - cek SSID/password");

    println!("Sesudah connect_async()");
    println!("Wi-Fi connected");

    // ========================================================================
    // 11. WAIT DHCP
    // ========================================================================

    println!("Menunggu DHCP...");

    stack.wait_config_up().await;

    // ========================================================================
    // 12. PRINT IP ADDRESS
    // ========================================================================

    let ip_config = stack
        .config_v4()
        .expect("config_v4 kosong setelah DHCP");

    println!(
        "IP address: {}",
        ip_config.address.address()
    );

    println!("DHCP OK");

    // ========================================================================
    // 13. SPAWN MQTT TASK
    // 14. SPAWN ACQUISITION TASK
    // ========================================================================

    spawner
        .spawn(mqtt_task(stack).expect("Gagal membuat mqtt_task"));

    spawner
        .spawn(
            acquisition_task(
                dht22_chamber,
                dht22_ambient,
            )
            .expect("Gagal membuat acquisition_task"),
        );

    // ========================================================================
    // 15. MAIN IDLE
    // ========================================================================

    loop {
        Timer::after(Duration::from_secs(3600)).await;
    }
}

// ============================================================================
// NETWORK TASK
// ============================================================================

/// Menjalankan network stack (Wi-Fi driver polling, DHCP, ARP, dst.).
///
/// Task ini tidak pernah selesai dan harus tetap berjalan
/// sepanjang program hidup.
#[embassy_executor::task]
async fn net_task(
    mut runner: Runner<'static, esp_radio::wifi::Interface<'static>>,
) -> ! {
    runner.run().await;
}

// ============================================================================
// ACQUISITION TASK
// ============================================================================

/// Acquisition dummy async:
///
/// - 500 sample
/// - interval 2 detik sesuai kode asli
/// - TANPA busy-wait
/// - gas1-8 masih menggunakan dummy_sensor_data()
/// - temperature/humidity menggunakan DHT22 fisik
///
/// Jika DHT22 gagal dibaca:
/// - error dicetak ke serial
/// - nilai terakhir yang valid tetap digunakan
/// - ESP32 tidak panic
///
/// DHT22 #1 = chamber / primary
/// DHT22 #2 = ambient / backup
#[embassy_executor::task]
async fn acquisition_task(
    mut dht22_chamber: Dht22<'static>,
    mut dht22_ambient: Dht22<'static>,
) {
    println!();
    println!("=== ACQUISITION START ===");

    // ========================================================================
    // LAST VALID DHT22 VALUES
    // ========================================================================

    // Nilai awal sebelum pembacaan DHT22 pertama yang berhasil.
    //
    // Ini placeholder, bukan klaim data sensor nyata.

    let mut last_temperature_chamber: f32 = 0.0;
    let mut last_humidity_chamber: f32 = 0.0;

    let mut last_temperature_ambient: f32 = 0.0;
    let mut last_humidity_ambient: f32 = 0.0;

    let mut dht_chamber_ever_succeeded = false;
    let mut dht_ambient_ever_succeeded = false;

    // ========================================================================
    // SAMPLE LOOP
    // ========================================================================

    for number in 1..=TOTAL_SAMPLES {
        // ====================================================================
        // DHT22 #1 — CHAMBER / PRIMARY
        // ====================================================================

        match dht22_chamber.read() {
            Ok(Dht22Reading {
                temperature_c,
                humidity_pct,
            }) => {
                last_temperature_chamber = temperature_c;
                last_humidity_chamber = humidity_pct;

                dht_chamber_ever_succeeded = true;

                println!(
                    "DHT22 CHAMBER OK: temperature={:.2} C, humidity={:.2} %",
                    temperature_c,
                    humidity_pct
                );
            }

            Err(e) => {
                println!(
                    "DHT22 CHAMBER gagal (sample={}): {:?}",
                    number,
                    e
                );

                // ============================================================
                // DHT22 #2 — AMBIENT / BACKUP
                // ============================================================

                match dht22_ambient.read() {
                    Ok(Dht22Reading {
                        temperature_c,
                        humidity_pct,
                    }) => {
                        last_temperature_ambient = temperature_c;
                        last_humidity_ambient = humidity_pct;

                        dht_ambient_ever_succeeded = true;

                        println!(
                            "DHT22 AMBIENT BACKUP OK: temperature={:.2} C, humidity={:.2} %",
                            temperature_c,
                            humidity_pct
                        );
                    }

                    Err(backup_error) => {
                        println!(
                            "DHT22 AMBIENT BACKUP gagal (sample={}): {:?}",
                            number,
                            backup_error
                        );

                        if dht_chamber_ever_succeeded {
                            println!(
                                "Pakai nilai DHT22 primary terakhir yang valid"
                            );
                        } else if dht_ambient_ever_succeeded {
                            println!(
                                "Pakai nilai DHT22 ambient terakhir yang valid"
                            );
                        } else {
                            println!(
                                "Belum ada pembacaan DHT22 valid, pakai 0.0"
                            );
                        }
                    }
                }
            }
        }

        // ====================================================================
        // GAS SENSOR — MASIH DUMMY
        // ====================================================================

        let gas_dummy = dummy_sensor_data().gas;

        // ====================================================================
        // BUILD SAMPLE
        // ====================================================================

        let sample = Sample {
            number,

            data: SensorData {
                gas: gas_dummy,

                // DHT22 #1 / GPIO10 = DATASET UTAMA
                temperature: last_temperature_chamber,
                humidity: last_humidity_chamber,
            },
        };

        // ====================================================================
        // SERIAL MONITOR
        // ====================================================================

        println!(
            "sample={}, \
             gas1={:.2}, gas2={:.2}, gas3={:.2}, gas4={:.2}, \
             gas5={:.2}, gas6={:.2}, gas7={:.2}, gas8={:.2}, \
             chamber_temp={:.2}, chamber_humidity={:.2}, \
             ambient_temp={:.2}, ambient_humidity={:.2}",
            sample.number,
            sample.data.gas[0],
            sample.data.gas[1],
            sample.data.gas[2],
            sample.data.gas[3],
            sample.data.gas[4],
            sample.data.gas[5],
            sample.data.gas[6],
            sample.data.gas[7],
            last_temperature_chamber,
            last_humidity_chamber,
            last_temperature_ambient,
            last_humidity_ambient,
        );

        // ====================================================================
        // SEND SAMPLE TO MQTT TASK
        // ====================================================================

        SAMPLE_CHANNEL.send(sample).await;

        // ====================================================================
        // SAMPLING INTERVAL
        // ====================================================================

        Timer::after(Duration::from_secs(2)).await;
    }

    println!("=== ACQUISITION FINISHED ===");
}

// ============================================================================
// JSON PAYLOAD
// ============================================================================

/// Membangun payload JSON satu sample ke buffer fixed-size
/// (`heapless::String`).
///
/// Tidak menggunakan heap allocation dan tidak menggunakan serde_json.
fn build_json_payload(
    sample: &Sample,
    out: &mut HString<200>,
) -> Result<(), core::fmt::Error> {
    out.clear();

    write!(
        out,
        "{{\"sample\":{},\
        \"gas1\":{:.2},\
        \"gas2\":{:.2},\
        \"gas3\":{:.2},\
        \"gas4\":{:.2},\
        \"gas5\":{:.2},\
        \"gas6\":{:.2},\
        \"gas7\":{:.2},\
        \"gas8\":{:.2},\
        \"temperature\":{:.2},\
        \"humidity\":{:.2}}}",
        sample.number,
        sample.data.gas[0],
        sample.data.gas[1],
        sample.data.gas[2],
        sample.data.gas[3],
        sample.data.gas[4],
        sample.data.gas[5],
        sample.data.gas[6],
        sample.data.gas[7],
        sample.data.temperature,
        sample.data.humidity,
    )
}

// ============================================================================
// MQTT TASK
// ============================================================================

/// Resolve DNS lalu connect ke ThingsBoard Cloud.
///
/// Sample diterima dari acquisition_task melalui Channel lalu
/// dipublish ke ThingsBoard.
///
/// Jika koneksi gagal:
/// - tampilkan error
/// - tunggu 3 detik
/// - resolve DNS ulang
/// - reconnect
#[embassy_executor::task]
async fn mqtt_task(
    stack: embassy_net::Stack<'static>,
) -> ! {
    println!();
    println!("=== MQTT TASK START ===");

    let mut json_buf: HString<200> = HString::new();

    // ========================================================================
    // RECONNECT LOOP
    // ========================================================================

    loop {
        // ====================================================================
        // DNS RESOLVE
        // ====================================================================

        println!();
        println!("DNS resolving {}...", TB_HOST);

        let resolved = match stack
            .dns_query(TB_HOST, DnsQueryType::A)
            .await
        {
            Ok(addrs) => addrs,

            Err(e) => {
                println!(
                    "DNS resolve gagal: {:?} -- coba lagi dalam 3 detik",
                    e
                );

                Timer::after(Duration::from_secs(3)).await;

                continue;
            }
        };

        // ====================================================================
        // AMBIL IP HASIL DNS
        // ====================================================================

        let broker_addr: IpAddress = match resolved.first() {
            Some(addr) => *addr,

            None => {
                println!(
                    "DNS resolve tidak mengembalikan alamat apa pun \
                     -- coba lagi dalam 3 detik"
                );

                Timer::after(Duration::from_secs(3)).await;

                continue;
            }
        };

        println!("DNS resolved");

        // ====================================================================
        // TCP CONNECT
        // ====================================================================

        println!("Connecting to ThingsBoard...");

        let mut rx_buffer = [0u8; 2048];
        let mut tx_buffer = [0u8; 2048];

        let mut socket = TcpSocket::new(
            stack,
            &mut rx_buffer,
            &mut tx_buffer,
        );

        socket.set_timeout(Some(Duration::from_secs(10)));

        if let Err(e) = socket
            .connect((broker_addr, TB_PORT))
            .await
        {
            println!(
                "MQTT TCP connect gagal: {:?} -- coba lagi dalam 3 detik",
                e
            );

            Timer::after(Duration::from_secs(3)).await;

            continue;
        }

        println!("MQTT TCP connected");

        // ====================================================================
        // MQTT CLIENT
        // ====================================================================

        let mut client = MqttClient::new(&mut socket);

        let connect_options = ConnectOptions {
            username: Some(TB_ACCESS_TOKEN),
            password: None,
            last_will: None,
        };

        // ====================================================================
        // MQTT CONNECT
        // ====================================================================

        if let Err(e) = client
            .connect_with_options(
                MQTT_CLIENT_ID,
                60,
                &connect_options,
            )
            .await
        {
            println!(
                "MQTT CONNECT gagal: {:?} -- coba lagi dalam 3 detik",
                e
            );

            Timer::after(Duration::from_secs(3)).await;

            continue;
        }

        println!("MQTT connected");

        // ====================================================================
        // PUBLISH LOOP
        // ====================================================================

        // Selama koneksi masih hidup, terima sample dari acquisition_task
        // dan publish ke ThingsBoard.

        loop {
            let sample = SAMPLE_CHANNEL.receive().await;

            // ================================================================
            // BUILD JSON
            // ================================================================

            if build_json_payload(
                &sample,
                &mut json_buf,
            )
            .is_err()
            {
                println!(
                    "Gagal membangun payload JSON untuk sample={}",
                    sample.number
                );

                continue;
            }

            // ================================================================
            // MQTT PUBLISH
            // ================================================================

            match client
                .publish(
                    TB_TOPIC,
                    json_buf.as_bytes(),
                )
                .await
            {
                Ok(()) => {
                    println!(
                        "MQTT publish OK: sample={}",
                        sample.number
                    );
                }

                Err(e) => {
                    println!(
                        "MQTT publish gagal (sample={}): {:?} -- reconnect",
                        sample.number,
                        e
                    );

                    break;
                }
            }
        }

        // ====================================================================
        // RECONNECT DELAY
        // ====================================================================

        Timer::after(Duration::from_secs(3)).await;
    }
}