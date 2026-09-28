use rumqttc::{
    Client,
    MqttOptions,
    QoS,
    TlsConfiguration,
    Transport,
    Connection,
};
use std::time::Duration;

pub fn connect_mqtt() -> (Client, Connection) {
    let host = "fcf83b0182fc4a57a327374bbeee936b.s1.eu.hivemq.cloud";
    let username = "geyynab";
    let password = "nabillaputri041004";

    let mut mqttoptions = MqttOptions::new(
        "rust-backend",
        host,
        8883,
    );

    mqttoptions.set_credentials(username, password);

    mqttoptions.set_transport(
        Transport::Tls(TlsConfiguration::default())
    );

    mqttoptions.set_keep_alive(
        Duration::from_secs(30)
    );

    let (client, connection) =
        Client::new(mqttoptions, 10);

    println!("Mencoba terhubung ke HiveMQ...");

    (client, connection)
}

pub fn subscribe_to_class_a(
    client: &Client,
) -> Result<(), rumqttc::ClientError> {
    client.subscribe(
        "classA/data",
        QoS::AtLeastOnce,
    )?;

    println!(
        "Berhasil subscribe ke classA/data!"
    );

    Ok(())
}

pub fn publish_ota_start(
    client: &Client,
    version: &str,
    file_size: usize,
    chunk_size: usize,
) -> Result<(), rumqttc::ClientError> {

    let payload = format!(
        r#"{{
            "type": "ota_start",
            "version": "{}",
            "size": {},
            "chunk_size": {}
        }}"#,
        version,
        file_size,
        chunk_size
    );

    client.publish(
        "classA/ota",
        QoS::AtLeastOnce,
        false,
        payload,
    )?;

    println!("================================");
    println!("OTA START berhasil dikirim!");
    println!("Topic   : classA/ota");
    println!("Version : {}", version);
    println!("Size    : {} bytes", file_size);
    println!("Chunk   : {} bytes", chunk_size);
    println!("================================");

    Ok(())
}

pub fn publish_firmware_chunks(
    client: &Client,
    firmware_path: &str,
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {

    let firmware =
        std::fs::read(firmware_path)?;

    let total_size = firmware.len();

    let total_chunks =
        (total_size + chunk_size - 1)
            / chunk_size;

    println!("================================");
    println!("MULAI KIRIM FIRMWARE");
    println!("File          : {}", firmware_path);
    println!("Ukuran        : {} bytes", total_size);
    println!("Chunk size    : {} bytes", chunk_size);
    println!("Total chunks  : {}", total_chunks);
    println!("================================");

    for (index, chunk) in
        firmware.chunks(chunk_size).enumerate()
    {

        client.publish(
            "classA/ota/data",
            QoS::AtLeastOnce,
            false,
            chunk,
        )?;

        println!(
            "📡 Chunk {}/{} terkirim ({} bytes)",
            index + 1,
            total_chunks,
            chunk.len()
        );
    }

    println!("================================");
    println!("SEMUA CHUNK FIRMWARE TERKIRIM");
    println!("================================");

    Ok(())
}

pub fn publish_ota_end(
    client: &Client,
    version: &str,
) -> Result<(), rumqttc::ClientError> {

    let payload = format!(
        r#"{{
            "type": "ota_end",
            "version": "{}"
        }}"#,
        version
    );

    client.publish(
        "classA/ota",
        QoS::AtLeastOnce,
        false,
        payload,
    )?;

    println!("================================");
    println!("OTA END berhasil dikirim!");
    println!("Topic   : classA/ota");
    println!("Version : {}", version);
    println!("================================");

    Ok(())
}