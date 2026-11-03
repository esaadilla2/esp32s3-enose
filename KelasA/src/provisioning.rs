
// src/provisioning.rs
// Wi-Fi credential storage using ESP-IDF compatible NVS.

extern crate alloc;

use alloc::string::String;

use esp_nvs::{Key, Nvs};
use esp_println::println;
use esp_storage::FlashStorage;

use heapless::String as HString;

/// Maximum lengths for SSID and password.
pub const MAX_SSID_LEN: usize = 32;
pub const MAX_PASS_LEN: usize = 64;

/// NVS partition from partitions.csv.
pub const CRED_FLASH_OFFSET: usize = 0x9000;
pub const CRED_FLASH_SIZE: usize = 0x4000;

/// NVS namespace and keys.
const WIFI_NAMESPACE: Key = Key::from_str("wifi_config");
const SSID_KEY: Key = Key::from_str("ssid");
const PASS_KEY: Key = Key::from_str("password");

/// Save Wi-Fi credentials to NVS.
/// Returns true on success.
pub fn save_credentials(
    flash: &mut FlashStorage<'static>,
    ssid: &HString<32>,
    pass: &HString<64>,
) -> bool {
    if ssid.is_empty()
        || ssid.len() > MAX_SSID_LEN
        || pass.len() > MAX_PASS_LEN
    {
        println!("[PROVISIONING] Invalid credential length.");
        return false;
    }

    let mut nvs = match Nvs::new(
        CRED_FLASH_OFFSET,
        CRED_FLASH_SIZE,
        &mut *flash,
    ) {
        Ok(nvs) => nvs,
        Err(_) => {
            println!("[PROVISIONING] Failed to open NVS.");
            return false;
        }
    };

    let result = nvs
        .set(&WIFI_NAMESPACE, &SSID_KEY, ssid.as_str())
        .and_then(|_| {
            nvs.set(&WIFI_NAMESPACE, &PASS_KEY, pass.as_str())
        });

    drop(nvs);

    match result {
        Ok(()) => {
            println!("[PROVISIONING] Wi-Fi credentials saved.");
            true
        }
        Err(_) => {
            println!("[PROVISIONING] Failed to save credentials.");
            false
        }
    }
}

/// Load Wi-Fi credentials from NVS.
/// Returns None when credentials are unavailable.
pub fn load_credentials(
    flash: &mut FlashStorage<'static>,
) -> Option<(HString<32>, HString<64>)> {
    let mut nvs = match Nvs::new(
        CRED_FLASH_OFFSET,
        CRED_FLASH_SIZE,
        &mut *flash,
    ) {
        Ok(nvs) => nvs,
        Err(_) => {
            println!("[PROVISIONING] NVS unavailable or empty.");
            return None;
        }
    };

    let ssid: String = match nvs.get(&WIFI_NAMESPACE, &SSID_KEY) {
        Ok(value) => value,
        Err(_) => {
            println!("[PROVISIONING] SSID not found in NVS.");
            return None;
        }
    };

    let pass: String = match nvs.get(&WIFI_NAMESPACE, &PASS_KEY) {
        Ok(value) => value,
        Err(_) => {
            println!("[PROVISIONING] Password not found in NVS.");
            return None;
        }
    };

    drop(nvs);

    if ssid.is_empty()
        || ssid.len() > MAX_SSID_LEN
        || pass.len() > MAX_PASS_LEN
    {
        println!("[PROVISIONING] Invalid credentials in NVS.");
        return None;
    }

    let mut ssid_out = HString::<32>::new();
    let mut pass_out = HString::<64>::new();

    ssid_out.push_str(&ssid).ok()?;
    pass_out.push_str(&pass).ok()?;

    println!("[PROVISIONING] Wi-Fi credentials loaded from NVS.");

    Some((ssid_out, pass_out))
}

/// Provisioning task placeholder.
pub async fn provisioning_task() {
    println!(
        "[PROVISIONING] NVS storage is ready. \
         Wi-Fi provisioning interface is not implemented yet."
    );
}
