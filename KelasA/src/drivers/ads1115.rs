//! Driver ADS1115 (16-bit ADC, I2C) minimal untuk ESP32-S3.
//!
//! Mendukung DUA instance ADS1115 (alamat berbeda) di SATU bus I2C yang
//! sama. Driver ini TIDAK menyimpan bus I2C di dalam struct -- setiap
//! method menerima `&mut I2C` sebagai parameter, supaya satu bus I2C yang
//! dibuat di `main.rs` bisa dipinjamkan bergantian ke kedua instance
//! `Ads1115` tanpa perlu crate bus-sharing tambahan.
//!
//! Generik di atas `embedded_hal::i2c::I2c` (trait blocking standar),
//! BUKAN terikat ke tipe konkret `esp_hal::i2c::master::I2c` -- driver
//! ini dikonfirmasi kompatibel karena esp-hal 1.1.0 mengimplementasikan
//! trait embedded-hal untuk `I2c` (lihat dokumentasi resmi modul
//! `esp_hal::i2c::master`).
//!
//! PENTING (batasan yang harus dipahami sebelum dipakai):
//! - Driver ini HANYA menangani komunikasi I2C + konversi raw ADC -> volt.
//!   Sinyal sensor gas 0-5V TIDAK aman masuk langsung ke ADS1115/ESP32
//!   tanpa pembagi tegangan/level shifting di sisi hardware -- itu di
//!   luar cakupan file ini, dan driver ini TIDAK mengklaim sudah aman.
//! - TIDAK ada konversi ke ppm/konsentrasi gas apa pun di sini.
//! - Konfigurasi PGA (lihat `PGA_FULL_SCALE_VOLTS`) dan data rate adalah
//!   ASUMSI DEFAULT yang wajar untuk pembacaan umum -- BUKAN nilai yang
//!   dikonfirmasi dari spesifikasi hardware Anda. Lihat catatan di bawah
//!   struct `Config` untuk detail dan alasan.

#![allow(dead_code)]

use embedded_hal::i2c::I2c;

// ============================================================================
// REGISTER & KONSTANTA ADS1115
// ============================================================================

/// Alamat register Conversion (hasil ADC 16-bit).
const REG_CONVERSION: u8 = 0x00;
/// Alamat register Config.
const REG_CONFIG: u8 = 0x01;
// Register Lo_thresh (0x02) dan Hi_thresh (0x03) TIDAK dipakai karena
// comparator dinonaktifkan (lihat CFG_COMP_QUE_DISABLE) -- driver ini
// tidak memakai pin ALERT/RDY.

// --- Bit-bit Config Register (16-bit), MSB dikirim/diterima duluan ---

/// OS (bit 15): tulis 1 untuk mulai single conversion (mode single-shot).
const CFG_OS_START_SINGLE: u16 = 1 << 15;
/// OS (bit 15) saat DIBACA: 1 = conversion sudah selesai / siap, 0 = masih berjalan.
const CFG_OS_READY_MASK: u16 = 1 << 15;

/// MUX (bit 14:12): pembacaan single-ended AIN0..AIN3 terhadap GND,
/// sesuai tabel yang Anda berikan.
const CFG_MUX_AIN0: u16 = 0b100 << 12;
const CFG_MUX_AIN1: u16 = 0b101 << 12;
const CFG_MUX_AIN2: u16 = 0b110 << 12;
const CFG_MUX_AIN3: u16 = 0b111 << 12;

/// Pengaturan Programmable Gain Amplifier (PGA) ADS1115 (bit 11:9 Config Register).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pga {
    /// FSR +-6.144 V (bit 11:9 = 000). Resolusi nominal = 6.144 / 32768 ~ 0.1875 mV/bit.
    /// Dicantumkan dalam dokumen pemetaan hardware jika ADS1115 ditenagai VDD = 5V.
    /// PERINGATAN: Sesuai datasheet Section 7.1, tegangan analog input TIDAK BOLEH
    /// melebihi VDD + 0.3V meskipun FSR disetel ke +-6.144V!
    Fsr6_144V,
    /// FSR +-4.096 V (bit 11:9 = 001). Resolusi nominal = 4.096 / 32768 = 0.1250 mV/bit.
    /// Pilihan default yang aman untuk rentang sinyal <= 3.3V / <= 4.096V.
    Fsr4_096V,
}

impl Pga {
    pub const fn config_bits(self) -> u16 {
        match self {
            Pga::Fsr6_144V => 0b000 << 9,
            Pga::Fsr4_096V => 0b001 << 9,
        }
    }

    pub const fn full_scale_volts(self) -> f32 {
        match self {
            Pga::Fsr6_144V => 6.144,
            Pga::Fsr4_096V => 4.096,
        }
    }

    pub fn raw_to_voltage(self, raw: i16) -> f32 {
        (raw as f32) * (self.full_scale_volts() / 32768.0)
    }
}

/// PGA default saat ini: +-4.096V (dapat diubah ke Fsr6_144V via with_pga()
/// setelah konfigurasi VDD dan pembagi tegangan hardware dikonfirmasi).
pub const DEFAULT_PGA: Pga = Pga::Fsr4_096V;

/// MODE (bit 8) = 1 -> single-shot (sesuai instruksi, bukan continuous).
const CFG_MODE_SINGLE_SHOT: u16 = 1 << 8;

/// DR (bit 7:5) = 100 -> 128 samples/detik (nilai default datasheet ADS1115).
///
/// ASUMSI: dipilih karena tidak ada persyaratan data rate spesifik di
/// instruksi Anda. 128 SPS cukup cepat untuk 8 channel dibaca bergantian
/// dalam siklus akuisisi 1 Hz, dengan margin noise yang wajar.
const CFG_DR_128SPS: u16 = 0b100 << 5;

/// Comparator dinonaktifkan sepenuhnya: COMP_MODE=0, COMP_POL=0,
/// COMP_LAT=0 (default 0, tidak perlu di-OR eksplisit), COMP_QUE[1:0]=11.
/// Ini adalah pengaturan yang direkomendasikan datasheet ADS1115 ketika
/// pin ALERT/RDY tidak dipakai.
const CFG_COMP_QUE_DISABLE: u16 = 0b11;

/// Jumlah maksimum percobaan polling saat menunggu conversion selesai,
/// sebelum dianggap timeout. Tidak memakai delay tambahan (tidak ada
/// dependency Delay di driver ini) -- setiap iterasi polling sudah makan
/// waktu nyata karena berupa transaksi I2C sungguhan.
///
/// ANALISIS TIMING: Pada 128 SPS, konversi internal ADS1115 memakan waktu ~7,81 ms
/// (dapat mencapai ~8,6 ms dengan toleransi osilator internal +-10%).
/// Transaksi I2C polling membaca 2 byte status:
/// - Pada 100 kHz: ~0,29 ms per poll -> 20 poll hanya ~5,8 ms (TIMEOUT PREMATUR!).
///   Dengan 150 poll (~43,5 ms budget timeout), conversion siap di poll ~27 (~7,8 ms).
/// - Pada 400 kHz: ~0,08 ms per poll -> 20 poll hanya ~1,6 ms (TIMEOUT PREMATUR!).
///   Dengan 150 poll (~12 ms budget timeout), conversion siap di poll ~98 (~7,8 ms).
///
/// Begitu bit OS siap (conversion-ready), loop langsung break, sehingga saat normal
/// hanya memakan ~7,8 ms per kanal.
const MAX_POLL_ATTEMPTS: u8 = 150;

// ============================================================================
// TIPE PUBLIK
// ============================================================================

/// Channel input single-ended ADS1115 (AIN0..AIN3 terhadap GND).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Ain0,
    Ain1,
    Ain2,
    Ain3,
}

impl Channel {
    /// Konversi indeks 0..3 menjadi `Channel`. Mengembalikan `None` untuk
    /// indeks di luar itu -- dipakai oleh pemanggil yang perlu memetakan
    /// dari indeks numerik (mis. loop) dan tetap butuh error eksplisit
    /// untuk indeks tidak valid.
    pub fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Channel::Ain0),
            1 => Some(Channel::Ain1),
            2 => Some(Channel::Ain2),
            3 => Some(Channel::Ain3),
            _ => None,
        }
    }

    fn mux_bits(self) -> u16 {
        match self {
            Channel::Ain0 => CFG_MUX_AIN0,
            Channel::Ain1 => CFG_MUX_AIN1,
            Channel::Ain2 => CFG_MUX_AIN2,
            Channel::Ain3 => CFG_MUX_AIN3,
        }
    }
}

/// Hasil pembacaan 4 channel satu ADS1115, sudah dalam satuan Volt.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Ads1115Reading {
    pub ain0: f32,
    pub ain1: f32,
    pub ain2: f32,
    pub ain3: f32,
}

/// Error driver ADS1115. `E` adalah tipe error dari implementasi I2C yang
/// dipakai (mis. `esp_hal::i2c::master::Error` lewat trait embedded-hal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ads1115Error<E> {
    /// Kegagalan transaksi I2C (NACK, bus error, dll -- diteruskan apa
    /// adanya dari implementasi I2C).
    I2c(E),
    /// Indeks channel di luar 0..3.
    InvalidChannel,
    /// Conversion tidak selesai dalam `MAX_POLL_ATTEMPTS` kali polling.
    Timeout,
}

impl<E> From<E> for Ads1115Error<E> {
    fn from(err: E) -> Self {
        Ads1115Error::I2c(err)
    }
}

// ============================================================================
// DRIVER
// ============================================================================

/// Driver satu unit ADS1115, diidentifikasi oleh alamat I2C-nya.
/// TIDAK menyimpan bus I2C -- lihat catatan desain di atas file ini.
pub struct Ads1115 {
    address: u8,
    pga: Pga,
}

impl Ads1115 {
    /// Buat instance untuk satu ADS1115 pada alamat I2C tertentu
    /// (mis. 0x48 atau 0x49) dengan PGA default (DEFAULT_PGA = +-4.096V).
    /// Tidak melakukan transaksi I2C apa pun -- murni menyimpan alamat dan konfigurasi.
    pub fn new(address: u8) -> Self {
        Self {
            address,
            pga: DEFAULT_PGA,
        }
    }

    /// Pasang pengaturan PGA khusus (mis. Pga::Fsr6_144V jika ditenagai 5V).
    pub fn with_pga(mut self, pga: Pga) -> Self {
        self.pga = pga;
        self
    }

    /// Alamat I2C instance ini.
    pub fn address(&self) -> u8 {
        self.address
    }

    /// Pengaturan PGA instance ini.
    pub fn pga(&self) -> Pga {
        self.pga
    }

    /// Baca satu channel, kembalikan raw ADC 16-bit signed (i16).
    ///
    /// Langkah:
    /// 1. Tulis Config register (OS=start, MUX=channel, PGA, MODE=single-shot, DR, comparator off)
    /// 2. Polling Config register sampai bit OS kembali 1 (conversion selesai) atau timeout
    /// 3. Baca Conversion register, kembalikan sebagai i16
    pub fn read_channel_raw<I2C>(
        &self,
        i2c: &mut I2C,
        channel: Channel,
    ) -> Result<i16, Ads1115Error<I2C::Error>>
    where
        I2C: I2c,
    {
        let config: u16 = CFG_OS_START_SINGLE
            | channel.mux_bits()
            | self.pga.config_bits()
            | CFG_MODE_SINGLE_SHOT
            | CFG_DR_128SPS
            | CFG_COMP_QUE_DISABLE;

        let config_bytes = config.to_be_bytes();
        i2c.write(
            self.address,
            &[REG_CONFIG, config_bytes[0], config_bytes[1]],
        )?;

        // Polling Config register: pointer sudah menunjuk REG_CONFIG dari
        // penulisan di atas, jadi cukup `read` langsung (tanpa menulis
        // ulang pointer) untuk membaca config saat ini.
        let mut ready = false;
        for _ in 0..MAX_POLL_ATTEMPTS {
            let mut status_bytes = [0u8; 2];
            i2c.read(self.address, &mut status_bytes)?;
            let status = u16::from_be_bytes(status_bytes);
            if status & CFG_OS_READY_MASK != 0 {
                ready = true;
                break;
            }
        }
        if !ready {
            return Err(Ads1115Error::Timeout);
        }

        // Baca Conversion register (pointer diubah dulu ke REG_CONVERSION
        // lewat write_read, satu transaksi dengan repeated-start).
        let mut raw_bytes = [0u8; 2];
        i2c.write_read(self.address, &[REG_CONVERSION], &mut raw_bytes)?;
        Ok(i16::from_be_bytes(raw_bytes))
    }

    /// Sama seperti `read_channel_raw`, tapi langsung dikonversi ke Volt
    /// menggunakan FSR dari PGA instance ini.
    pub fn read_channel_voltage<I2C>(
        &self,
        i2c: &mut I2C,
        channel: Channel,
    ) -> Result<f32, Ads1115Error<I2C::Error>>
    where
        I2C: I2c,
    {
        let raw = self.read_channel_raw(i2c, channel)?;
        Ok(self.pga.raw_to_voltage(raw))
    }

    /// Baca AIN0..AIN3 satu ADS1115 ini secara berurutan (4 kali single-shot
    /// conversion), kembalikan sebagai `Ads1115Reading` dalam Volt.
    pub fn read_all<I2C>(&self, i2c: &mut I2C) -> Result<Ads1115Reading, Ads1115Error<I2C::Error>>
    where
        I2C: I2c,
    {
        Ok(Ads1115Reading {
            ain0: self.read_channel_voltage(i2c, Channel::Ain0)?,
            ain1: self.read_channel_voltage(i2c, Channel::Ain1)?,
            ain2: self.read_channel_voltage(i2c, Channel::Ain2)?,
            ain3: self.read_channel_voltage(i2c, Channel::Ain3)?,
        })
    }

    /// Baca satu channel berdasarkan indeks 0..3 (bukan `Channel` enum) --
    /// mengembalikan `Ads1115Error::InvalidChannel` untuk indeks di luar
    /// itu.
    pub fn read_channel_index_voltage<I2C>(
        &self,
        i2c: &mut I2C,
        index: u8,
    ) -> Result<f32, Ads1115Error<I2C::Error>>
    where
        I2C: I2c,
    {
        let channel = Channel::from_index(index).ok_or(Ads1115Error::InvalidChannel)?;
        self.read_channel_voltage(i2c, channel)
    }
}

/// Konversi raw ADC 16-bit (signed) ke Volt berdasarkan PGA tertentu.
pub fn raw_to_voltage_pga(raw: i16, pga: Pga) -> f32 {
    pga.raw_to_voltage(raw)
}

/// Konversi raw ADC 16-bit (signed) ke Volt, memakai full-scale range PGA default
/// (`DEFAULT_PGA` = +-4.096V).
///
/// TIDAK melakukan konversi ke ppm atau satuan konsentrasi gas apa pun --
/// hanya tegangan hasil ADC apa adanya.
pub fn raw_to_voltage(raw: i16) -> f32 {
    DEFAULT_PGA.raw_to_voltage(raw)
}
