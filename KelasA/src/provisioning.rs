// src/provisioning.rs
// Simple provisioning stub. In a real product this would start an AP and HTTP server
// to receive SSID/PASSWORD from a user. Here we just provide a placeholder function.

use embedded_storage::nor_flash::{NorFlash, ReadNorFlash};
use esp_println::println; // for debug logs

use esp_storage::FlashStorage;


/// Maximum lengths for SSID and password stored in flash.
pub const MAX_SSID_LEN: usize = 32;
pub const MAX_PASS_LEN: usize = 64;
/// Fixed flash offset for storing Wi‑Fi credentials. Choose an area not used by OTA.
pub const CRED_FLASH_OFFSET: usize = 0x9000; // use NVS partition area (safe region)

/// Save credentials to flash. Returns true on success.
use heapless::String as HString; // use heapless string for no_std

/// Save credentials to flash. Returns true on success.
pub fn save_credentials(flash: &mut FlashStorage<'static>, ssid: &HString<32>, pass: &HString<64>) -> bool {
    let mut buf = [0xFFu8; MAX_SSID_LEN + MAX_PASS_LEN];
    let ssid_bytes = ssid.as_bytes();
    let pass_bytes = pass.as_bytes();
    if ssid_bytes.len() > MAX_SSID_LEN || pass_bytes.len() > MAX_PASS_LEN {
        println!("Credentials too long for flash storage");
        return false;
    }
    buf[..ssid_bytes.len()].copy_from_slice(ssid_bytes);
    buf[MAX_SSID_LEN..MAX_SSID_LEN + pass_bytes.len()].copy_from_slice(pass_bytes);
    flash
        .erase(CRED_FLASH_OFFSET as u32, (buf.len()) as u32)
        .ok()
        .and_then(|_| flash.write(CRED_FLASH_OFFSET as u32, &buf).ok())
        .is_some()
}

/// Load credentials from flash. Returns (ssid, password) if present.
pub fn load_credentials(flash: &mut FlashStorage<'static>) -> Option<(heapless::String<32>, heapless::String<64>)> {
    let mut buf = [0xFFu8; MAX_SSID_LEN + MAX_PASS_LEN];
    if flash.read(CRED_FLASH_OFFSET as u32, &mut buf).is_err() {
        return None;
    }
    if buf.iter().all(|&b| b == 0xFF) {
        return None;
    }
    let ssid_len = buf.iter().position(|&b| b == 0x00).unwrap_or(MAX_SSID_LEN);
    let pass_start = MAX_SSID_LEN;
    let pass_len = buf[pass_start..]
        .iter()
        .position(|&b| b == 0x00)
        .unwrap_or(MAX_PASS_LEN);
    let mut ssid = heapless::String::<32>::new();
    ssid.push_str(core::str::from_utf8(&buf[..ssid_len]).ok()?).ok()?;
    let mut pass = heapless::String::<64>::new();
    pass.push_str(core::str::from_utf8(&buf[pass_start..pass_start + pass_len]).ok()?).ok()?;
    Some((ssid, pass))
}


/// Provisioning task placeholder – in real use this would spin up an AP and web UI.
pub async fn provisioning_task() {
    println!("[PROVISIONING] No Wi‑Fi credentials stored. Provisioning not implemented in this stub.");
    // User can flash credentials via the "save_credentials" helper from host side.
}
