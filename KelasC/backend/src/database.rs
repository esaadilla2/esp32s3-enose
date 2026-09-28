use rusqlite::{Connection, Result};

pub fn init_database() -> Result<Connection> {
    let db = Connection::open("dashboard.db")?;

    db.execute(
        "CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL
        )",
        [],
    )?;

    println!("Database SQLite siap!");

    Ok(db)
}

pub fn save_report(
    db: &Connection,
    device_id: &str,
    timestamp: &str,
    classification: &str,
    confidence: f64,
) -> Result<()> {
    db.execute(
        "INSERT INTO reports
        (device_id, timestamp, classification, confidence)
        VALUES (?1, ?2, ?3, ?4)",
        (
            device_id,
            timestamp,
            classification,
            confidence,
        ),
    )?;

    println!("Data berhasil disimpan ke database!");

    Ok(())
}