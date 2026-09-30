# ESP32-S3 E-Nose Firmware

Firmware perangkat **Electronic Nose (E-Nose)** berbasis **ESP32-S3** yang dikembangkan menggunakan **Rust**.

## Scope

Pengembangan firmware mencakup:

- Sensor gas menggunakan 2× ADS1115
- DHT22
- Data acquisition dan sampling
- MQTT communication
- Persiapan TinyML
- Persiapan OTA

## Current Progress

Firmware saat ini telah mencakup:

- ESP32-S3 firmware dasar
- Wi-Fi connectivity
- MQTT telemetry
- Pengujian DHT22
- Driver ADS1115 untuk dua device dengan alamat I2C `0x48` dan `0x49`
- Struktur data acquisition dan sampling

Integrasi ADS1115 dengan hardware pengukuran utama, timestamp final, TinyML, dan OTA masih dalam tahap pengembangan.

## Project Structure

```text 
src/
├── bin/
│   └── main.rs
├── drivers/
│   ├── ads1115.rs
│   ├── dht22.rs
│   └── mod.rs
├── acquisition.rs
├── config.rs
├── lib.rs
└── logger.rs
```
## Documentation

Dokumentasi lengkap mengenai perkembangan implementasi dan pengujian proyek dapat dilihat pada:

**[Laporan Progress ↗](https://drive.google.com/file/d/1faPqN9mEQJcEoNNcXS-VtY8iLRyIR7xJ/view?usp=sharing)**

## Development

| Component | Technology |
|---|---|
| Language | Rust |
| Target | ESP32-S3 |
| Framework | `esp-hal` + Embassy |
| Build System | Cargo |

## Check

```bash
cargo check
```

##Build 

```bash
cargo build
```

