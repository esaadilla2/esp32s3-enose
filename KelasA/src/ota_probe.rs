//! OTA TAHAP A -- pemeriksaan BACA-SAJA (tidak menulis/menghapus flash, tidak mengaktifkan slot).
//!
//! STATUS: BELUM DIKOMPILASI. Dibuat tanpa akses source esp-storage 0.9 / esp-bootloader-esp-idf 0.5,
//! hanya dari daftar API yang kamu berikan. Semua baris bertanda `VERIFIKASI` harus dicocokkan
//! dengan source lokal (`cargo doc --open -p esp-bootloader-esp-idf` atau baca ~/.cargo/registry/src/*).
//!
//! Pemakaian (sementara, di main() SETELAH esp_hal::init, sebelum spawn task):
//!     esp32s3_test::ota_probe::inspect(peripherals.FLASH);
//! Daftarkan di src/lib.rs: `pub mod ota_probe;`  (file ini disalin sebagai src/ota_probe.rs)

use esp_bootloader_esp_idf::ota_updater::OtaUpdater;      // VERIFIKASI path modul (bisa `esp_bootloader_esp_idf::ota::...`)
use esp_bootloader_esp_idf::partitions::PARTITION_TABLE_MAX_LEN; // VERIFIKASI nama konstanta
use esp_println::println;
use esp_storage::FlashStorage;

pub fn inspect(flash_peripheral: esp_hal::peripherals::FLASH<'static>) {
    // VERIFIKASI: di esp-storage 0.9 konstruktor bisa `FlashStorage::new(FLASH)` (butuh peripheral)
    // atau `FlashStorage::new()`; sesuaikan dengan source.
    let mut flash = FlashStorage::new(flash_peripheral);

    // Buffer baca tabel partisi: PERIKSA tabel nyata di perangkat, bukan asumsi partitions.csv.
    let mut buf = [0u8; PARTITION_TABLE_MAX_LEN];
    let mut updater = match OtaUpdater::new(&mut flash, &mut buf) {
        Ok(u) => u,
        Err(e) => {
            println!("OTA: tabel partisi di perangkat TIDAK mendukung OTA / gagal dibaca: {:?}", e);
            return; // JANGAN lanjut ke Tahap B kalau ini terjadi
        }
    };

    // Hanya BACA otadata & state. Tidak ada set_current_ota_state / activate_next_partition di sini.
    match updater.selected_partition() {
        Ok(p) => println!("OTA: slot berjalan = {:?}", p),   // VERIFIKASI: tipe kembalian (AppPartitionSubType?)
        Err(e) => println!("OTA: selected_partition gagal: {:?}", e),
    }
    match updater.current_ota_state() {
        Ok(s) => println!("OTA: state = {:?}", s),
        Err(e) => println!("OTA: current_ota_state gagal: {:?}", e),
    }
    // next_partition() hanya MENGHITUNG slot tujuan (mengembalikan FlashRegion); jangan write/erase di Tahap A.
    match updater.next_partition() {
        Ok((region, subtype)) => {
            // VERIFIKASI: bentuk tuple & cara mengambil ukuran (mis. `region.capacity()` dari trait embedded-storage).
            println!("OTA: slot tujuan = {:?}", subtype);
            let _ = region;
        }
        Err(e) => println!("OTA: next_partition gagal: {:?}", e),
    }
    println!("OTA: pemeriksaan selesai (tanpa tulis/aktivasi).");
}
