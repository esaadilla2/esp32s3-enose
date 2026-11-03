
use std::fs;
use std::path::PathBuf;

fn main() {
    // Generate partitions.bin secara otomatis
    generate_partition_bin();

    // Fungsi linker bawaan project
    linker_be_nice();

    // Pastikan linkall.x menjadi linker script terakhir.
    println!("cargo:rustc-link-arg=-Tlinkall.x");
}

// Membuat partitions.bin dari partitions.csv
fn generate_partition_bin() {
    println!("cargo:rerun-if-changed=partitions.csv");

    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR tidak ditemukan"),
    );

    let csv_path = manifest_dir.join("partitions.csv");
    let bin_path = manifest_dir.join("partitions.bin");

    // Membaca file CSV
    let csv = fs::read_to_string(&csv_path)
        .expect("Gagal membaca partitions.csv");

    // Mengubah CSV menjadi partition table
    let table = esp_idf_part::PartitionTable::try_from_str(csv)
        .expect("Format partitions.csv tidak valid");

    // Memvalidasi partition table
    table.validate()
        .expect("Partition table tidak valid");

    // Mengubah partition table menjadi binary
    let binary = table.to_bin()
        .expect("Gagal mengubah partition table menjadi binary");

    // Menyimpan hasil sebagai partitions.bin
    fs::write(&bin_path, binary)
        .expect("Gagal membuat partitions.bin");

    println!("cargo:warning=partitions.bin berhasil dibuat!");
}

fn linker_be_nice() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        let kind = &args[1];
        let what = &args[2];

        match kind.as_str() {
            "undefined-symbol" => match what.as_str() {
                what if what.starts_with("_defmt_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `defmt` not found - make sure `defmt.x` is added as a linker script and you have included `use defmt_rtt as _;`"
                    );
                    eprintln!();
                }
                "_stack_start" => {
                    eprintln!();
                    eprintln!("💡 Is the linker script `linkall.x` missing?");
                    eprintln!();
                }
                what if what.starts_with("esp_rtos_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `esp-radio` has no scheduler enabled. Make sure you have initialized `esp-rtos` or provided an external scheduler."
                    );
                    eprintln!();
                }
                "embedded_test_linker_file_not_added_to_rustflags" => {
                    eprintln!();
                    eprintln!(
                        "💡 `embedded-test` not found - make sure `embedded-test.x` is added as a linker script for tests"
                    );
                    eprintln!();
                }
                "free"
                | "malloc"
                | "calloc"
                | "get_free_internal_heap_size"
                | "malloc_internal"
                | "realloc_internal"
                | "calloc_internal"
                | "free_internal" => {
                    eprintln!();
                    eprintln!(
                        "💡 Did you forget the `esp-alloc` dependency or didn't enable the `compat` feature on it?"
                    );
                    eprintln!();
                }
                _ => (),
            },
            // Tidak ada pesan khusus untuk "missing-lib".
            _ => {
                std::process::exit(1);
            }
        }

        std::process::exit(0);
    }

    // GNU --error-handling-script sengaja tidak digunakan
    // karena linker yang dipakai adalah rust-lld.
}
