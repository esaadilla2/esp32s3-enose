use axum::extract::{Multipart, State};
use axum::Json;
use axum::http::header;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use rusqlite::Connection;
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize)]
pub struct ReportResponse {
    pub id: i64,
    pub device_id: String,
    pub timestamp: String,
    pub classification: String,
    pub confidence: f64,
}

pub type Db = Arc<Mutex<Connection>>;

pub async fn get_reports(
    State(db): State<Db>,
) -> Json<Vec<ReportResponse>> {
    let db = db.lock().unwrap();

    let mut statement = db
        .prepare(
            "SELECT id, device_id, timestamp, classification, confidence
             FROM reports
             ORDER BY id DESC",
        )
        .unwrap();

    let reports = statement
        .query_map([], |row| {
            Ok(ReportResponse {
                id: row.get(0)?,
                device_id: row.get(1)?,
                timestamp: row.get(2)?,
                classification: row.get(3)?,
                confidence: row.get(4)?,
            })
        })
        .unwrap();

    let mut result = Vec::new();

    for report in reports {
        result.push(report.unwrap());
    }

    Json(result)
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub active_devices: i64,
    pub total_records: i64,
    pub average_confidence: f64,
}

pub async fn get_stats(
    State(db): State<Db>,
) -> Json<StatsResponse> {
    let db = db.lock().unwrap();

    let total_records: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM reports",
            [],
            |row| row.get(0),
        )
        .unwrap();

    let active_devices: i64 = db
        .query_row(
            "SELECT COUNT(DISTINCT device_id) FROM reports",
            [],
            |row| row.get(0),
        )
        .unwrap();

    let average_confidence: f64 = db
        .query_row(
            "SELECT COALESCE(AVG(confidence), 0) FROM reports",
            [],
            |row| row.get(0),
        )
        .unwrap();

    Json(StatsResponse {
        active_devices,
        total_records,
        average_confidence,
    })
}

pub async fn generate_pdf(
    State(db): State<Db>,
) -> impl IntoResponse {

    // Ambil akses database
    let db = db.lock().unwrap();

    // Generate PDF
    if let Err(error) =
        crate::pdf::generate_report_pdf(&db)
    {
        println!("❌ Gagal membuat PDF: {:?}", error);

        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Gagal membuat PDF",
        )
            .into_response();
    }

    // Lepaskan database lock
    drop(db);

    // Baca file PDF
    let pdf = match std::fs::read(
        "classification_report.pdf"
    ) {
        Ok(file) => file,

        Err(error) => {
            println!(
                "Gagal membaca PDF: {:?}",
                error
            );

            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Gagal membaca file PDF",
            )
                .into_response();
        }
    };

    (
        StatusCode::OK,
        [
            (
                header::CONTENT_TYPE,
                "application/pdf",
            ),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"classification_report.pdf\"",
            ),
        ],
        pdf,
    )
        .into_response()
}

pub async fn upload_firmware(
    mut multipart: Multipart,
) -> impl IntoResponse {

    while let Ok(Some(field)) = multipart.next_field().await {

        let field_name = field.name().unwrap_or("");

        if field_name != "firmware" {
            continue;
        }

        let data = match field.bytes().await {
            Ok(data) => data,
            Err(error) => {
                println!(
                    "Gagal membaca file firmware: {:?}",
                    error
                );

                return (
                    StatusCode::BAD_REQUEST,
                    "Gagal membaca file firmware",
                )
                    .into_response();
            }
        };

        // Membuat folder firmware kalau belum ada
        if let Err(error) =
            std::fs::create_dir_all("firmware")
        {
            println!(
                "Gagal membuat folder firmware: {:?}",
                error
            );

            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Gagal membuat folder firmware",
            )
                .into_response();
        }

        // Simpan firmware
        if let Err(error) =
            std::fs::write(
                "firmware/latest.bin",
                &data,
            )
        {
            println!(
                "Gagal menyimpan firmware: {:?}",
                error
            );

            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Gagal menyimpan firmware",
            )
                .into_response();
        }

        println!("================================");
        println!("Firmware berhasil diupload!");
        println!("Ukuran: {} bytes", data.len());
        println!("Lokasi: firmware/latest.bin");
        println!("================================");

        return (
            StatusCode::OK,
            format!(
                "Firmware berhasil diupload! Ukuran: {} bytes",
                data.len()
            ),
        )
            .into_response();
    }

    (
        StatusCode::BAD_REQUEST,
        "File firmware tidak ditemukan",
    )
        .into_response()
}