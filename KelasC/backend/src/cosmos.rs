use base64::{engine::general_purpose, Engine as _};
use hmac::{Hmac, Mac};
use reqwest::Client;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

fn generate_authorization(
    verb: &str,
    resource_type: &str,
    resource_link: &str,
    date: &str,
    key: &str,
) -> String {
    let payload = format!(
        "{}\n{}\n{}\n{}\n\n",
        verb.to_lowercase(),
        resource_type.to_lowercase(),
        resource_link,
        date.to_lowercase()
    );

    let key_bytes = general_purpose::STANDARD
        .decode(key)
        .expect("COSMOS_KEY bukan Base64 yang valid");

    let mut mac =
        HmacSha256::new_from_slice(&key_bytes)
            .expect("Gagal membuat HMAC");

    mac.update(payload.as_bytes());

    let signature =
        general_purpose::STANDARD
            .encode(mac.finalize().into_bytes());

    let auth = format!(
        "type=master&ver=1.0&sig={}",
        signature
    );

    urlencoding::encode(&auth).to_string()
}


pub async fn save_report(
    device_id: &str,
    timestamp: &str,
    classification: &str,
    confidence: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::var("COSMOS_ENDPOINT")?;
    let key = std::env::var("COSMOS_KEY")?;
    let database = std::env::var("COSMOS_DATABASE")?;
    let container = std::env::var("COSMOS_CONTAINER")?;

    let client = Client::new();

    let date = chrono::Utc::now()
        .format("%a, %d %b %Y %H:%M:%S GMT")
        .to_string();

    let resource_link =
        format!("dbs/{}/colls/{}", database, container);

    let authorization = generate_authorization(
        "POST",
        "docs",
        &resource_link,
        &date,
        &key,
    );

    let url = format!(
        "{}/{}/docs",
        endpoint.trim_end_matches('/'),
        resource_link
    );

    let item = serde_json::json!({
        "id": format!("{}-{}", device_id, chrono::Utc::now().timestamp_millis()),
        "device_id": device_id,
        "timestamp": timestamp,
        "classification": classification,
        "confidence": confidence
    });

    let response = client
        .post(&url)
        .header("Authorization", authorization)
        .header("x-ms-date", &date)
        .header("x-ms-version", "2018-12-31")
        .header("Content-Type", "application/json")
        .header(
            "x-ms-documentdb-partitionkey",
            format!(r#"["{}"]"#, device_id),
        )
        .json(&item)
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let body = response.text().await?;
        return Err(
            format!("Cosmos DB gagal: {} - {}", status, body).into()
        );
    }

    println!(
        "Report tersimpan ke Cosmos DB | device={} classification={} confidence={}",
        device_id,
        classification,
        confidence
    );

    Ok(())
}