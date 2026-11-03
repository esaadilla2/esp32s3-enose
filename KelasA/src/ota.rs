//! OTA via ThingsBoard Cloud -- bagian perangkat (otadata, slot, reboot).
//!
//! Protokol MQTT + penulisan/verifikasi flash ada di `ota_core.rs`. File ini hanya:
//!   - membaca/menulis state OTA lewat `esp-bootloader-esp-idf` (`OtaUpdater`),
//!   - memilih slot nonaktif (`next_partition`) dan mengaktifkannya SETELAH image terverifikasi,
//!   - melaporkan status ke ThingsBoard dan me-reboot.
//!
//! STATUS: BELUM DIKOMPILASI di mesin ini (tidak ada toolchain Xtensa). Pola pemakaian
//! `OtaUpdater`/`FlashStorage` sama dengan ota.rs lama yang sudah terbukti jalan di perangkat.

use embassy_net::tcp::TcpSocket;
use embassy_time::{Duration, Timer};
use embassy_mqtt_lite::MqttClient;
use esp_bootloader_esp_idf::ota::OtaImageState;
use esp_bootloader_esp_idf::ota_updater::OtaUpdater;
use esp_bootloader_esp_idf::partitions::PARTITION_TABLE_MAX_LEN;
use esp_println::println;
use esp_storage::FlashStorage;

use crate::ota_core::{self, OtaError, Query};

// ============================================================================
// IDENTITAS FIRMWARE -- UBAH UNTUK SETIAP VERSI
// ============================================================================

/// Harus SAMA PERSIS dengan "Title" paket firmware di ThingsBoard.
pub const FW_TITLE: &str = "coffee-enose";
/// Harus SAMA PERSIS dengan "Version" paket firmware di ThingsBoard.
/// Naikkan setiap build baru (mis. "2" untuk firmware awal, "3" untuk firmware baru).
pub const FW_VERSION: &str = "3";

/// Tipe socket yang dipakai client MQTT (juga dipakai oleh ota_core).
pub type Sock<'a> = TcpSocket<'a>;

/// Hasil pemeriksaan OTA di boot ini.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Firmware sudah sama dengan yang ditetapkan di cloud (atau tidak ada yang ditetapkan).
    UpToDate,
    /// Pemeriksaan/pembaruan gagal. Firmware lama tetap berjalan, slot baru tidak diaktifkan.
    Failed(OtaError),
}

pub struct OtaDevice {
    flash: FlashStorage<'static>,
    /// True bila boot ini adalah boot pertama setelah update (state otadata New/PendingVerify).
    just_updated: bool,
}

impl OtaDevice {
    /// Ambil alih peripheral FLASH dan baca state OTA boot ini (hanya membaca).
    pub fn new(flash_periph: esp_hal::peripherals::FLASH<'static>) -> Self {
        let mut dev = Self {
            flash: FlashStorage::new(flash_periph),
            just_updated: false,
        };
        dev.read_boot_state();
        dev
    }

    fn read_boot_state(&mut self) {
        let mut pt = [0u8; PARTITION_TABLE_MAX_LEN];
        let mut updater = match OtaUpdater::new(&mut self.flash, &mut pt) {
            Ok(u) => u,
            Err(e) => {
                println!("OTA: layout partisi tidak valid: {:?}", e);
                return;
            }
        };
        match updater.selected_partition() {
            Ok(p) => println!("OTA: slot terpilih di otadata = {:?}", p),
            Err(e) => println!("OTA: selected_partition gagal: {:?}", e),
        }
        match updater.current_ota_state() {
            Ok(s) => {
                // New = bootloader TIDAK mengubahnya (rollback otomatis tidak aktif).
                // PendingVerify = bootloader mengaktifkan rollback dan menunggu konfirmasi kita.
                println!("OTA: state saat boot = {:?}", s);
                self.just_updated =
                    matches!(s, OtaImageState::New | OtaImageState::PendingVerify);
            }
            Err(e) => println!("OTA: current_ota_state gagal: {:?}", e),
        }
    }

    /// Tandai image yang sedang berjalan VALID (boot sehat: Wi-Fi + MQTT berhasil).
    fn confirm_valid(&mut self) {
        if !self.just_updated {
            return;
        }
        let mut pt = [0u8; PARTITION_TABLE_MAX_LEN];
        let Ok(mut updater) = OtaUpdater::new(&mut self.flash, &mut pt) else {
            return;
        };
        match updater.set_current_ota_state(OtaImageState::Valid) {
            Ok(()) => println!("OTA: image yang berjalan dikonfirmasi VALID"),
            Err(e) => println!("OTA: gagal set state Valid: {:?}", e),
        }
        self.just_updated = false;
    }

    /// Pemeriksaan OTA satu kali per boot. Bila ada firmware baru: unduh, verifikasi,
    /// aktifkan slot, reboot (fungsi ini tidak kembali). Bila tidak: kembali dengan Outcome.
    ///
    /// Harus dipanggil tepat setelah MQTT CONNECT berhasil, SEBELUM sesi akuisisi dimulai
    /// (menulis flash dan reboot akan mengganggu pengukuran yang sedang berjalan).
    pub async fn check_and_update(
        &mut self,
        client: &mut MqttClient<'_, Sock<'_>>,
        rid: u32,
    ) -> Outcome {
        let just = self.just_updated;

        // Laporkan versi yang sedang berjalan.
        if let Err(e) = ota_core::report(client, FW_TITLE, FW_VERSION, None, None).await {
            return Outcome::Failed(e);
        }

        let query = ota_core::query_offer(client, FW_TITLE, FW_VERSION, rid).await;
        let offer = match query {
            Err(e) => {
                println!("OTA: pemeriksaan cloud gagal ({})", e.as_str());
                self.confirm_valid(); // boot ini sehat; cloud saja yang tidak terjangkau
                return Outcome::Failed(e);
            }
            Ok(Query::NoFirmware) => {
                println!("OTA: cloud tidak menetapkan firmware untuk perangkat ini");
                if just {
                    let _ = ota_core::report(client, FW_TITLE, FW_VERSION, Some("UPDATED"), None).await;
                }
                self.confirm_valid();
                return Outcome::UpToDate;
            }
            Ok(Query::UpToDate) => {
                println!("OTA: firmware sudah terbaru ({} {})", FW_TITLE, FW_VERSION);
                if just {
                    // Perangkat baru saja di-update dan cloud mengakui versinya cocok.
                    let _ = ota_core::report(client, FW_TITLE, FW_VERSION, Some("UPDATED"), None).await;
                    println!("OTA: update selesai, status UPDATED dilaporkan");
                }
                self.confirm_valid();
                return Outcome::UpToDate;
            }
            Ok(Query::Offer(o)) => o,
        };

        println!(
            "OTA: cloud menawarkan {} versi {} (berjalan: {})",
            offer.title.as_str(),
            offer.version.as_str(),
            FW_VERSION
        );

        if just {
            // Baru saja di-update tetapi cloud masih menginginkan versi lain: image yang
            // terpasang berlabel versi salah. Jangan mengunduh lagi (mencegah putaran update).
            println!("OTA: versi terpasang != versi di cloud setelah update -- TIDAK mengunduh lagi");
            let e = OtaError::VersionMismatch;
            let _ = ota_core::report(client, FW_TITLE, FW_VERSION, Some("FAILED"), Some(e.as_str())).await;
            self.confirm_valid();
            return Outcome::Failed(e);
        }

        // Image yang sedang berjalan sehat -> tandai valid SEBELUM menulis slot lain.
        self.confirm_valid();

        match self.install(client, &offer, rid).await {
            Ok(()) => {
                println!("OTA: slot baru AKTIF. Reboot dalam 1 detik...");
                Timer::after(Duration::from_secs(1)).await;
                esp_hal::system::software_reset()
            }
            Err(e) => {
                println!("OTA: GAGAL ({}) -- slot TIDAK diaktifkan, firmware lama tetap berjalan", e.as_str());
                let _ = ota_core::report(client, FW_TITLE, FW_VERSION, Some("FAILED"), Some(e.as_str())).await;
                Outcome::Failed(e)
            }
        }
    }

    /// Tulis ke slot nonaktif, verifikasi, lalu aktifkan. Aktivasi HANYA terjadi bila semua
    /// verifikasi di `fetch_firmware` lolos.
    async fn install(
        &mut self,
        client: &mut MqttClient<'_, Sock<'_>>,
        offer: &ota_core::Offer,
        rid: u32,
    ) -> Result<(), OtaError> {
        let mut pt = [0u8; PARTITION_TABLE_MAX_LEN];
        let mut updater = OtaUpdater::new(&mut self.flash, &mut pt).map_err(|e| {
            println!("OTA: OtaUpdater::new gagal: {:?}", e);
            OtaError::Flash
        })?;

        let res = {
            let (mut region, subtype) = updater.next_partition().map_err(|e| {
                println!("OTA: next_partition gagal: {:?}", e);
                OtaError::Flash
            })?;
            println!("OTA: menulis versi {} ke slot {:?}", offer.version.as_str(), subtype);
            ota_core::fetch_firmware(client, &mut region, FW_TITLE, FW_VERSION, offer, rid).await
        }; // region dilepas di sini
        res?;

        // Semua verifikasi lolos -> baru aktifkan.
        let _ = ota_core::report(client, FW_TITLE, FW_VERSION, Some("UPDATING"), None).await;
        updater.activate_next_partition().map_err(|e| {
            println!("OTA: activate_next_partition gagal: {:?}", e);
            OtaError::Flash
        })?;
        // State berlaku untuk slot yang BARU dipilih: New = menunggu konfirmasi boot pertama.
        updater
            .set_current_ota_state(OtaImageState::New)
            .map_err(|e| {
                println!("OTA: set state New gagal: {:?}", e);
                OtaError::Flash
            })?;
        Ok(())
    }
}
