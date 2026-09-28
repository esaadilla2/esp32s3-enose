use base64::{engine::general_purpose, Engine as _};
use hmac::{Hmac, Mac};
use rumqttc::{
    Client,
    Connection,
    MqttOptions,
    QoS,
    TlsConfiguration,
    Transport,
};
use sha2::Sha256;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

fn generate_sas_token(
    host: &str,
    device_id: &str,
    device_key: &str,
) -> String {

    let expiry =
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;

    let resource_uri =
        format!(
            "{}/devices/{}",
            host,
            device_id
        );

    let encoded_resource =
        urlencoding::encode(&resource_uri);

    let string_to_sign =
        format!(
            "{}\n{}",
            encoded_resource,
            expiry
        );

    let key =
        general_purpose::STANDARD
            .decode(device_key)
            .expect("AZURE_DEVICE_KEY bukan Base64 yang valid");

    let mut mac =
        HmacSha256::new_from_slice(&key)
            .expect("Gagal membuat HMAC");

    mac.update(
        string_to_sign.as_bytes()
    );

    let signature =
        general_purpose::STANDARD
            .encode(mac.finalize().into_bytes());

    let encoded_signature =
        urlencoding::encode(&signature);

    format!(
        "SharedAccessSignature sr={}&sig={}&se={}",
        encoded_resource,
        encoded_signature,
        expiry
    )
}

pub fn connect_azure_mqtt(
    host: &str,
    device_id: &str,
    device_key: &str,
) -> (Client, Connection) {

    let sas_token =
        generate_sas_token(
            host,
            device_id,
            device_key,
        );

    let mut mqttoptions =
        MqttOptions::new(
            device_id,
            host,
            8883,
        );

    let username =
        format!(
            "{}/{}/?api-version=2021-04-12",
            host,
            device_id
        );

    mqttoptions.set_credentials(
        username,
        sas_token,
    );

    mqttoptions.set_transport(
        Transport::Tls(
            TlsConfiguration::default()
        )
    );

    mqttoptions.set_keep_alive(
        Duration::from_secs(30)
    );

    let (client, connection) =
        Client::new(
            mqttoptions,
            10
        );

    println!("================================");
    println!("Azure IoT Hub MQTT siap");
    println!("Host   : {}", host);
    println!("Device : {}", device_id);
    println!("================================");

    (client, connection)
}

pub fn publish_test(
    client: &Client,
    device_id: &str,
) -> Result<(), rumqttc::ClientError> {

    let topic =
        format!(
            "devices/{}/messages/events/",
            device_id
        );

    let payload = r#"{
        "device_id": "ESP32-001",
        "classification": "AMAN",
        "confidence": 0.95
    }"#;

    client.publish(
        topic,
        QoS::AtLeastOnce,
        false,
        payload,
    )?;

    println!("================================");
    println!("Test message dikirim ke Azure!");
    println!("================================");

    Ok(())
}