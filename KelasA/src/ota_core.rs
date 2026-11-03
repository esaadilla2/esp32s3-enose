//! OTA via ThingsBoard MQTT -- INTI PORTABEL (protokol + penulisan + verifikasi).
//!
//! File ini sengaja TIDAK memakai crate esp-hal / esp-bootloader / esp-storage, supaya
//! logikanya bisa diuji di PC (lihat harness uji di panduan). Bagian yang menyentuh
//! hardware (otadata, aktivasi slot, reboot) ada di `ota.rs`.
//!
//! Protokol (ThingsBoard MQTT Device API, firmware):
//!   1. subscribe  v1/devices/me/attributes/response/+
//!   2. publish    v1/devices/me/attributes/request/{rid}
//!                 {"sharedKeys":"fw_title,fw_version,fw_size,fw_checksum,fw_checksum_algorithm"}
//!   3. terima     v1/devices/me/attributes/response/{rid}   (JSON atribut shared)
//!   4. subscribe  v2/fw/response/+/chunk/+
//!   5. publish    v2/fw/request/{rid}/chunk/{idx}  payload = ukuran chunk (ASCII)
//!   6. terima     v2/fw/response/{rid}/chunk/{idx} payload = byte firmware
//!                 idx naik satu per permintaan; chunk kosong = akhir firmware.
//!   Status ke cloud: telemetry v1/devices/me/telemetry
//!                 {"current_fw_title","current_fw_version","fw_state","fw_error"}.
//!
//! Catatan penting tentang embassy-mqtt-lite: `receive()` TIDAK aman dibatalkan di tengah
//! paket. Karena itu setiap error/timeout di sini dianggap fatal untuk update: slot tidak
//! diaktifkan dan pemanggil harus menutup koneksi lalu menyambung ulang.

use core::fmt::Write as _;

use embassy_mqtt_lite::MqttClient;
use embassy_time::{Duration, with_timeout};
use embedded_storage::nor_flash::{NorFlash, ReadNorFlash};
use esp_println::println;
use heapless::String as HString;
use sha2::{Digest, Sha256};

use crate::ota::Sock;

// ============================================================================
// KONSTANTA
// ============================================================================

/// Ukuran chunk yang diminta. Sama dengan ukuran sektor flash supaya satu chunk = satu sektor.
pub const CHUNK_SIZE: usize = 4096;
const SECTOR: usize = 4096;
/// Buffer terima: chunk + topic + header MQTT.
const RX_BUF: usize = CHUNK_SIZE + 128;

const IMAGE_MAGIC: u8 = 0xE9;
const CHIP_ID_ESP32S3: u16 = 9;
const HDR_LEN: usize = 24;

const ATTR_TIMEOUT: Duration = Duration::from_secs(10);
const CHUNK_TIMEOUT: Duration = Duration::from_secs(10);
/// Permintaan ulang per chunk bila tidak ada jawaban (selain permintaan pertama).
const CHUNK_RETRIES: u8 = 2;

const TOPIC_TELEMETRY: &str = "v1/devices/me/telemetry";
const TOPIC_ATTR_RESP_FILTER: &str = "v1/devices/me/attributes/response/+";
const TOPIC_CHUNK_RESP_FILTER: &str = "v2/fw/response/+/chunk/+";
const SHARED_KEYS: &str =
    "{\"sharedKeys\":\"fw_title,fw_version,fw_size,fw_checksum,fw_checksum_algorithm\"}";

// ============================================================================
// ERROR
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtaError {
    /// Error MQTT/TCP (koneksi harus dianggap rusak).
    Mqtt,
    /// Tidak ada jawaban dalam batas waktu.
    Timeout,
    /// Atribut firmware dari cloud tidak lengkap/tidak valid.
    BadOffer,
    /// fw_title paket berbeda dari judul firmware ini.
    TitleMismatch,
    /// Algoritma checksum selain SHA-256.
    UnsupportedChecksum,
    /// Firmware lebih besar dari slot OTA.
    TooLarge,
    /// Chunk datang di luar urutan / ukuran chunk tidak valid.
    ChunkOrder,
    /// Jumlah byte tidak sama dengan fw_size.
    SizeMismatch,
    /// Header/trailer image bukan image aplikasi ESP32-S3 yang valid.
    BadImage,
    /// Erase/write/read flash gagal.
    Flash,
    /// SHA-256 tidak cocok dengan fw_checksum.
    ChecksumMismatch,
    /// Setelah update, cloud masih menawarkan versi yang berbeda dari yang berjalan.
    VersionMismatch,
}

impl OtaError {
    /// Nilai untuk telemetry `fw_error` (huruf kecil, tanpa karakter yang perlu di-escape).
    pub fn as_str(&self) -> &'static str {
        match self {
            OtaError::Mqtt => "mqtt_error",
            OtaError::Timeout => "timeout",
            OtaError::BadOffer => "bad_offer",
            OtaError::TitleMismatch => "title_mismatch",
            OtaError::UnsupportedChecksum => "unsupported_checksum",
            OtaError::TooLarge => "too_large",
            OtaError::ChunkOrder => "chunk_order",
            OtaError::SizeMismatch => "size_mismatch",
            OtaError::BadImage => "bad_image",
            OtaError::Flash => "flash_error",
            OtaError::ChecksumMismatch => "checksum_mismatch",
            OtaError::VersionMismatch => "version_mismatch",
        }
    }

    /// True bila koneksi MQTT kemungkinan sudah tidak sinkron dan harus disambung ulang.
    pub fn connection_suspect(&self) -> bool {
        matches!(
            self,
            OtaError::Mqtt | OtaError::Timeout | OtaError::ChunkOrder
        )
    }
}

fn mqtt_err<E: core::fmt::Debug>(what: &str, e: E) -> OtaError {
    println!("OTA: {} gagal: {:?}", what, e);
    OtaError::Mqtt
}

// ============================================================================
// TELEMETRY STATUS
// ============================================================================

/// Kirim identitas firmware yang berjalan + (opsional) fw_state / fw_error.
pub async fn report(
    client: &mut MqttClient<'_, Sock<'_>>,
    title: &str,
    version: &str,
    state: Option<&str>,
    error: Option<&str>,
) -> Result<(), OtaError> {
    let mut j: HString<256> = HString::new();
    let ok = write!(
        j,
        "{{\"current_fw_title\":\"{}\",\"current_fw_version\":\"{}\"",
        title, version
    )
    .is_ok()
        && state.map_or(true, |s| write!(j, ",\"fw_state\":\"{}\"", s).is_ok())
        && error.map_or(true, |e| write!(j, ",\"fw_error\":\"{}\"", e).is_ok())
        && j.push('}').is_ok();
    if !ok {
        return Err(OtaError::BadOffer);
    }
    client
        .publish(TOPIC_TELEMETRY, j.as_bytes())
        .await
        .map_err(|e| mqtt_err("publish status", e))
}

// ============================================================================
// ATRIBUT FIRMWARE (shared attributes)
// ============================================================================

pub struct Offer {
    pub title: HString<48>,
    pub version: HString<32>,
    /// fw_size dari cloud. None bila tidak dikirim (maka akhir ditentukan chunk kosong).
    pub size: Option<usize>,
    pub hash: [u8; 32],
}

pub enum Query {
    /// Tidak ada firmware yang ditetapkan untuk perangkat ini.
    NoFirmware,
    /// Judul + versi sama dengan yang berjalan.
    UpToDate,
    /// Ada firmware lain yang harus dipasang.
    Offer(Offer),
}

/// Cari `"key"` diikuti `:` dan kembalikan teks nilai mentah (setelah spasi).
fn json_raw<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let mut from = 0;
    while let Some(rel) = s[from..].find(key) {
        let start = from + rel;
        let end = start + key.len();
        let quoted = start > 0
            && s.as_bytes()[start - 1] == b'"'
            && s.as_bytes().get(end) == Some(&b'"');
        if quoted {
            let rest = s[end + 1..].trim_start();
            if let Some(v) = rest.strip_prefix(':') {
                return Some(v.trim_start());
            }
        }
        from = end;
    }
    None
}

fn json_str<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let v = json_raw(s, key)?.strip_prefix('"')?;
    let end = v.find('"')?;
    let out = &v[..end];
    if out.contains('\\') { None } else { Some(out) }
}

fn json_usize(s: &str, key: &str) -> Option<usize> {
    let v = json_raw(s, key)?;
    let v = v.strip_prefix('"').unwrap_or(v);
    let digits = v.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    v[..digits].parse::<usize>().ok()
}

fn nib(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = (nib(b[2 * i])? << 4) | nib(b[2 * i + 1])?;
    }
    Some(out)
}

fn alg_is_sha256(s: &str) -> bool {
    let mut norm: HString<16> = HString::new();
    for c in s.chars().filter(|c| c.is_ascii_alphanumeric()) {
        if norm.push(c.to_ascii_uppercase()).is_err() {
            return false;
        }
    }
    norm.as_str() == "SHA256"
}

/// Terjemahkan JSON jawaban atribut menjadi keputusan. Fungsi murni (mudah diuji).
pub fn parse_offer(json: &str, own_title: &str, own_version: &str) -> Result<Query, OtaError> {
    let (title, version) = match (json_str(json, "fw_title"), json_str(json, "fw_version")) {
        (Some(t), Some(v)) => (t, v),
        _ => return Ok(Query::NoFirmware),
    };
    if title != own_title {
        println!("OTA: judul paket \"{}\" != judul firmware ini \"{}\"", title, own_title);
        return Err(OtaError::TitleMismatch);
    }
    if version == own_version {
        return Ok(Query::UpToDate);
    }
    // Checksum: hanya SHA-256 (64 digit hex). Algoritma yang tidak disebut dianggap SHA-256
    // (default ThingsBoard), tetapi panjang hex tetap harus 64.
    if let Some(alg) = json_str(json, "fw_checksum_algorithm") {
        if !alg_is_sha256(alg) {
            println!("OTA: algoritma checksum \"{}\" tidak didukung (hanya SHA-256)", alg);
            return Err(OtaError::UnsupportedChecksum);
        }
    }
    let hash = json_str(json, "fw_checksum")
        .and_then(parse_hex32)
        .ok_or_else(|| {
            println!("OTA: fw_checksum tidak ada / bukan SHA-256 (64 hex)");
            OtaError::BadOffer
        })?;
    let mut t: HString<48> = HString::new();
    let mut v: HString<32> = HString::new();
    t.push_str(title).map_err(|_| OtaError::BadOffer)?;
    v.push_str(version).map_err(|_| OtaError::BadOffer)?;
    Ok(Query::Offer(Offer {
        title: t,
        version: v,
        size: json_usize(json, "fw_size"),
        hash,
    }))
}

/// Minta atribut shared firmware ke ThingsBoard dan putuskan apa yang harus dilakukan.
pub async fn query_offer(
    client: &mut MqttClient<'_, Sock<'_>>,
    own_title: &str,
    own_version: &str,
    rid: u32,
) -> Result<Query, OtaError> {
    client
        .subscribe(TOPIC_ATTR_RESP_FILTER)
        .await
        .map_err(|e| mqtt_err("subscribe attributes/response", e))?;

    let mut req: HString<64> = HString::new();
    let mut want: HString<64> = HString::new();
    write!(req, "v1/devices/me/attributes/request/{}", rid).map_err(|_| OtaError::BadOffer)?;
    write!(want, "v1/devices/me/attributes/response/{}", rid).map_err(|_| OtaError::BadOffer)?;

    client
        .publish(req.as_str(), SHARED_KEYS.as_bytes())
        .await
        .map_err(|e| mqtt_err("publish attributes/request", e))?;

    let mut buf = [0u8; 1024];
    loop {
        let msg = match with_timeout(ATTR_TIMEOUT, client.receive(&mut buf)).await {
            Err(_) => {
                println!("OTA: timeout menunggu atribut firmware");
                return Err(OtaError::Timeout);
            }
            Ok(Err(e)) => return Err(mqtt_err("receive attributes", e)),
            Ok(Ok(m)) => m,
        };
        if msg.topic != want.as_str() {
            continue;
        }
        let text = core::str::from_utf8(msg.payload).map_err(|_| OtaError::BadOffer)?;
        return parse_offer(text, own_title, own_version);
    }
}

// ============================================================================
// PENULIS FLASH (per sektor 4 KiB: erase -> write) + hash stream
// ============================================================================

struct Writer<'a, R: NorFlash> {
    region: &'a mut R,
    sector: [u8; SECTOR],
    fill: usize,
    flash_off: usize,
    total: usize,
    hasher: Sha256,
}

impl<'a, R> Writer<'a, R>
where
    R: NorFlash,
    R::Error: core::fmt::Debug,
{
    fn new(region: &'a mut R) -> Self {
        Self {
            region,
            sector: [0xFF; SECTOR],
            fill: 0,
            flash_off: 0,
            total: 0,
            hasher: Sha256::new(),
        }
    }

    fn push(&mut self, mut data: &[u8]) -> Result<(), OtaError> {
        self.hasher.update(data);
        self.total += data.len();
        while !data.is_empty() {
            let n = (SECTOR - self.fill).min(data.len());
            self.sector[self.fill..self.fill + n].copy_from_slice(&data[..n]);
            self.fill += n;
            data = &data[n..];
            if self.fill == SECTOR {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), OtaError> {
        if self.fill == 0 {
            return Ok(());
        }
        if self.flash_off == 0 {
            // Header image aplikasi ESP: magic, jumlah segmen, chip id (LE di offset 12).
            let chip = u16::from_le_bytes([self.sector[12], self.sector[13]]);
            let segs = self.sector[1];
            if self.fill < HDR_LEN
                || self.sector[0] != IMAGE_MAGIC
                || chip != CHIP_ID_ESP32S3
                || segs == 0
                || segs > 16
            {
                println!(
                    "OTA: header image salah (magic=0x{:02X}, chip={}, segmen={})",
                    self.sector[0], chip, segs
                );
                return Err(OtaError::BadImage);
            }
        }
        let padded = (self.fill + 3) & !3;
        self.sector[self.fill..].fill(0xFF);

        let off = self.flash_off;
        self.region
            .erase(off as u32, (off + SECTOR) as u32)
            .map_err(|e| {
                println!("OTA: erase gagal @0x{:X}: {:?}", off, e);
                OtaError::Flash
            })?;
        self.region
            .write(off as u32, &self.sector[..padded])
            .map_err(|e| {
                println!("OTA: write gagal @0x{:X}: {:?}", off, e);
                OtaError::Flash
            })?;

        self.fill = 0;
        self.flash_off += SECTOR;
        Ok(())
    }

    /// Flush sisa dan kembalikan SHA-256 seluruh byte yang diterima.
    fn finish(mut self) -> Result<[u8; 32], OtaError> {
        self.flush()?;
        let h = self.hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&h);
        Ok(out)
    }
}

/// Baca ulang image dari flash: (1) SHA-256 seluruh isi == checksum dari cloud,
/// (2) bila header menandai "hash appended", SHA-256 internal image (32 byte terakhir)
/// harus cocok -- ini memeriksa image itu sendiri, bukan hanya transfernya.
fn verify_flash<R>(region: &mut R, size: usize, expect: &[u8; 32]) -> Result<(), OtaError>
where
    R: NorFlash,
    R::Error: core::fmt::Debug,
{
    let mut all = Sha256::new();
    let mut img = Sha256::new();
    let mut trailer = [0u8; 32];
    let mut hash_appended = 0u8;
    let img_end = size.saturating_sub(32);
    let mut buf = [0u8; SECTOR];
    let mut off = 0usize;
    while off < size {
        let n = (size - off).min(SECTOR);
        let padded = (n + 3) & !3;
        region.read(off as u32, &mut buf[..padded]).map_err(|e| {
            println!("OTA: baca-ulang gagal @0x{:X}: {:?}", off, e);
            OtaError::Flash
        })?;
        if off == 0 {
            if n < HDR_LEN {
                return Err(OtaError::BadImage);
            }
            hash_appended = buf[23];
        }
        all.update(&buf[..n]);
        let take = img_end.saturating_sub(off).min(n);
        img.update(&buf[..take]);
        let t_start = img_end.max(off);
        if t_start < off + n {
            let src = &buf[t_start - off..n];
            trailer[t_start - img_end..t_start - img_end + src.len()].copy_from_slice(src);
        }
        off += n;
    }

    if all.finalize().as_slice() != &expect[..] {
        return Err(OtaError::ChecksumMismatch);
    }
    if hash_appended == 1 {
        if size < HDR_LEN + 32 {
            return Err(OtaError::BadImage);
        }
        if img.finalize().as_slice() != &trailer[..] {
            println!("OTA: SHA-256 internal image tidak cocok (image rusak)");
            return Err(OtaError::BadImage);
        }
    }
    Ok(())
}

// ============================================================================
// UNDUH CHUNK
// ============================================================================

/// Terima satu chunk dengan indeks `idx`. Mengembalikan jumlah byte yang ditulis.
async fn recv_chunk<R>(
    client: &mut MqttClient<'_, Sock<'_>>,
    buf: &mut [u8],
    rid: u32,
    idx: u32,
    cap: usize,
    expected: Option<usize>,
    w: &mut Writer<'_, R>,
) -> Result<usize, OtaError>
where
    R: NorFlash,
    R::Error: core::fmt::Debug,
{
    loop {
        let msg = match with_timeout(CHUNK_TIMEOUT, client.receive(buf)).await {
            Err(_) => return Err(OtaError::Timeout),
            Ok(Err(e)) => return Err(mqtt_err("receive chunk", e)),
            Ok(Ok(m)) => m,
        };
        let Some(rest) = msg.topic.strip_prefix("v2/fw/response/") else {
            continue;
        };
        let Some((r, c)) = rest.split_once("/chunk/") else {
            return Err(OtaError::ChunkOrder);
        };
        let (Ok(got_rid), Ok(got_idx)) = (r.parse::<u32>(), c.parse::<u32>()) else {
            return Err(OtaError::ChunkOrder);
        };
        if got_rid != rid || got_idx > idx {
            println!("OTA: chunk tak terduga rid={} idx={} (minta idx={})", got_rid, got_idx, idx);
            return Err(OtaError::ChunkOrder);
        }
        if got_idx < idx {
            continue; // jawaban terlambat untuk permintaan ulang sebelumnya
        }
        let len = msg.payload.len();
        if len > CHUNK_SIZE {
            return Err(OtaError::ChunkOrder);
        }
        if w.total + len > cap {
            return Err(OtaError::TooLarge);
        }
        if let Some(sz) = expected {
            if w.total + len > sz {
                return Err(OtaError::SizeMismatch);
            }
        }
        if len > 0 {
            w.push(msg.payload)?;
        }
        return Ok(len);
    }
}

/// Unduh firmware dari ThingsBoard, tulis ke `region` (slot OTA nonaktif), verifikasi.
/// Slot BELUM diaktifkan di sini -- itu tugas pemanggil setelah fungsi ini mengembalikan Ok.
pub async fn fetch_firmware<R>(
    client: &mut MqttClient<'_, Sock<'_>>,
    region: &mut R,
    own_title: &str,
    own_version: &str,
    offer: &Offer,
    rid: u32,
) -> Result<(), OtaError>
where
    R: NorFlash,
    R::Error: core::fmt::Debug,
{
    let cap = region.capacity();
    if let Some(sz) = offer.size {
        if sz < HDR_LEN + 32 || sz > cap {
            println!("OTA: fw_size {} tidak muat di slot ({} byte)", sz, cap);
            return Err(OtaError::TooLarge);
        }
    }

    report(client, own_title, own_version, Some("DOWNLOADING"), None).await?;
    client
        .subscribe(TOPIC_CHUNK_RESP_FILTER)
        .await
        .map_err(|e| mqtt_err("subscribe v2/fw/response", e))?;

    let mut buf = [0u8; RX_BUF];
    let mut w = Writer::new(region);
    let mut idx: u32 = 0;
    let mut next_report = 64 * 1024;

    loop {
        if offer.size == Some(w.total) {
            break;
        }

        let mut topic: HString<64> = HString::new();
        let mut size_txt: HString<8> = HString::new();
        write!(topic, "v2/fw/request/{}/chunk/{}", rid, idx).map_err(|_| OtaError::ChunkOrder)?;
        write!(size_txt, "{}", CHUNK_SIZE).map_err(|_| OtaError::ChunkOrder)?;

        let mut attempt = 0u8;
        let len = loop {
            client
                .publish(topic.as_str(), size_txt.as_bytes())
                .await
                .map_err(|e| mqtt_err("publish chunk request", e))?;
            match recv_chunk(client, &mut buf, rid, idx, cap, offer.size, &mut w).await {
                Ok(n) => break n,
                Err(OtaError::Timeout) if attempt < CHUNK_RETRIES => {
                    attempt += 1;
                    println!("OTA: chunk {} timeout, minta ulang ({}/{})", idx, attempt, CHUNK_RETRIES);
                }
                Err(e) => return Err(e),
            }
        };

        if len == 0 {
            break; // chunk kosong = akhir firmware
        }
        idx += 1;
        if w.total >= next_report {
            println!("OTA: {} byte diterima", w.total);
            next_report += 64 * 1024;
        }
    }

    let total = w.total;
    if let Some(sz) = offer.size {
        if total != sz {
            println!("OTA: diterima {} byte, fw_size {}", total, sz);
            return Err(OtaError::SizeMismatch);
        }
    }
    if total < HDR_LEN + 32 {
        return Err(OtaError::BadImage);
    }

    let stream_hash = w.finish()?;
    if stream_hash != offer.hash {
        println!("OTA: SHA-256 unduhan tidak cocok dengan fw_checksum");
        return Err(OtaError::ChecksumMismatch);
    }
    println!("OTA: unduhan {} byte selesai, checksum cocok", total);
    report(client, own_title, own_version, Some("DOWNLOADED"), None).await?;

    verify_flash(region, total, &offer.hash)?;
    println!("OTA: baca-ulang dari flash OK (checksum + hash internal image)");
    report(client, own_title, own_version, Some("VERIFIED"), None).await?;
    Ok(())
}
