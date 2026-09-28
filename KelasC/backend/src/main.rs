mod api;
mod database;
mod pdf;
mod azure_mqtt;
mod cosmos;

use axum::{
    http::Method,
    routing::{get, post},
    Router
};
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};
use rustls::crypto::ring;

#[tokio::main]
async fn main() {
    ring::default_provider()
        .install_default()
        .expect("Gagal memasang Rustls CryptoProvider");

    dotenvy::dotenv().ok();
    let azure_host =
    std::env::var("AZURE_IOT_HOST")
        .expect("AZURE_IOT_HOST belum diatur");

    let azure_device_id =
        std::env::var("AZURE_DEVICE_ID")
            .expect("AZURE_DEVICE_ID belum diatur");

    let azure_device_key =
        std::env::var("AZURE_DEVICE_KEY")
            .expect("AZURE_DEVICE_KEY belum diatur");

    let (_azure_client, mut azure_connection) =
    azure_mqtt::connect_azure_mqtt(
        &azure_host,
        &azure_device_id,
        &azure_device_key,
    );

    std::thread::spawn(move || {
        for notification in azure_connection.iter() {
            match notification {
                Ok(event) => {
                    println!("Azure MQTT: {:?}", event);
                }
                Err(error) => {
                    println!("Azure MQTT error: {:?}", error);
                }
            }
        }
    });
        azure_mqtt::publish_test(
        &_azure_client,
        &azure_device_id,
    ).expect("Gagal publish ke Azure");

    println!("Azure MQTT client berhasil dibuat!");

    let cosmos_endpoint =
    std::env::var("COSMOS_ENDPOINT")
        .expect("COSMOS_ENDPOINT belum diatur");

    let cosmos_database =
        std::env::var("COSMOS_DATABASE")
            .expect("COSMOS_DATABASE belum diatur");

    let cosmos_container =
        std::env::var("COSMOS_CONTAINER")
            .expect("COSMOS_CONTAINER belum diatur");

    println!("================================");
    println!("Cosmos DB config terbaca");
    println!("Endpoint : {}", cosmos_endpoint);
    println!("Database : {}", cosmos_database);
    println!("Container: {}", cosmos_container);
    println!("================================");

    cosmos::save_report(
        "ESP32-001",
        "2026-09-28T22:00:00+07:00",
        "AMAN",
        0.95,
    )
    .await
    .expect("Gagal menyimpan report ke Cosmos DB");

    // DATABASE
    let db = database::init_database()
        .expect("Gagal membuka database");

    let db = Arc::new(Mutex::new(db));

    // API SERVER
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            ]);

    let app = Router::new()
        .route("/reports", get(api::get_reports))
        .route("/stats", get(api::get_stats))
        .route("/reports/pdf", get(api::generate_pdf))
        .route("/ota/upload", post(api::upload_firmware))
        .layer(cors)
        .with_state(db.clone());

    let listener = tokio::net::TcpListener::bind(
        "127.0.0.1:3000"
    )
    .await
    .expect("Gagal menjalankan server");

    println!("API berjalan di http://127.0.0.1:3000");
    println!("Endpoint: GET /reports");

    // JALANKAN API
    axum::serve(listener, app)
        .await
        .expect("Server berhenti");
}