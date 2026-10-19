use printpdf::{
    BuiltinFont,
    Color,
    Mm,
    Op,
    PdfDocument,
    PdfFontHandle,
    PdfPage,
    PdfSaveOptions,
    Point,
    Pt,
    Rgb,
    TextItem,
    TextRenderingMode,
};

use rusqlite::Connection;

pub fn generate_report_pdf(
    db: &Connection,
) -> Result<(), Box<dyn std::error::Error>> {

    println!("================================");
    println!("MEMBUAT PDF REPORT");
    println!("================================");

    // ==========================================
    // 1. AMBIL SEMUA DATA DARI SQLITE
    // ==========================================

    let mut statement = db.prepare(
        "SELECT id, device_id, timestamp,
                classification, confidence
         FROM reports
         ORDER BY id ASC",
    )?;

    let reports = statement.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, f64>(4)?,
        ))
    })?;

    let mut data = Vec::new();

    for report in reports {
        data.push(report?);
    }

    println!(
        "Jumlah data ditemukan: {}",
        data.len()
    );

    if data.is_empty() {
        return Err(
            "Belum ada data report di SQLite".into()
        );
    }

    // ==========================================
    // 2. BUAT DOCUMENT
    // ==========================================

    let mut doc = PdfDocument::new(
        "IoT Classification Report",
    );

    let mut pages: Vec<PdfPage> = Vec::new();
    let mut ops: Vec<Op> = Vec::new();

    // ==========================================
    // FUNGSI TEXT
    // ==========================================

    fn add_text(
        ops: &mut Vec<Op>,
        x: f32,
        y: f32,
        text: String,
        size: f32,
        bold: bool,
    ) {
        let font = if bold {
            BuiltinFont::HelveticaBold
        } else {
            BuiltinFont::Helvetica
        };

        ops.push(Op::StartTextSection);

        ops.push(Op::SetTextCursor {
            pos: Point::new(
                Mm(x),
                Mm(y),
            ),
        });

        ops.push(Op::SetFont {
            font: PdfFontHandle::Builtin(font),
            size: Pt(size),
        });

        ops.push(Op::SetFillColor {
            col: Color::Rgb(Rgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                icc_profile: None,
            }),
        });

        ops.push(Op::SetTextRenderingMode {
            mode: TextRenderingMode::Fill,
        });

        ops.push(Op::ShowText {
            items: vec![
                TextItem::Text(text)
            ],
        });

        ops.push(Op::EndTextSection);
    }

    // ==========================================
    // FUNGSI GARIS
    // ==========================================

    fn add_line(
        ops: &mut Vec<Op>,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) {
        use printpdf::{
            Line,
            LinePoint,
        };

        ops.push(Op::DrawLine {
            line: Line {
                points: vec![
                    LinePoint {
                        p: Point::new(
                            Mm(x1),
                            Mm(y1),
                        ),
                        bezier: false,
                    },
                    LinePoint {
                        p: Point::new(
                            Mm(x2),
                            Mm(y2),
                        ),
                        bezier: false,
                    },
                ],
                is_closed: false,
            },
        });
    }

    // ==========================================
    // HEADER HALAMAN
    // ==========================================

    fn add_header(
        ops: &mut Vec<Op>,
    ) -> f32 {

        // Judul
        add_text(
            ops,
            20.0,
            275.0,
            "IoT SMT 5 - Classification Report".to_string(),
            18.0,
            true,
        );

        // Subtitle
        add_text(
            ops,
            20.0,
            262.0,
            "ESP32-S3 Monitoring System".to_string(),
            11.0,
            false,
        );

        // ======================================
        // HEADER TABEL
        // ======================================

        let table_top = 245.0;

        // Garis atas
        add_line(
            ops,
            15.0,
            table_top + 5.0,
            195.0,
            table_top + 5.0,
        );

        // Nama kolom
        add_text(
            ops,
            18.0,
            table_top,
            "ID".to_string(),
            9.0,
            true,
        );

        add_text(
            ops,
            32.0,
            table_top,
            "Device".to_string(),
            9.0,
            true,
        );

        add_text(
            ops,
            68.0,
            table_top,
            "Timestamp".to_string(),
            9.0,
            true,
        );

        add_text(
            ops,
            125.0,
            table_top,
            "Classification".to_string(),
            9.0,
            true,
        );

        add_text(
            ops,
            172.0,
            table_top,
            "Confidence".to_string(),
            9.0,
            true,
        );

        // Garis bawah header
        add_line(
            ops,
            15.0,
            table_top - 5.0,
            195.0,
            table_top - 5.0,
        );

        table_top - 12.0
    }

    // ==========================================
    // HALAMAN PERTAMA
    // ==========================================

    let mut y = add_header(&mut ops);

    // ==========================================
    // DATA TABEL
    // ==========================================

    for (
        id,
        device_id,
        timestamp,
        classification,
        confidence,
    ) in &data {

        // --------------------------------------
        // CEK HALAMAN PENUH
        // --------------------------------------

        if y < 35.0 {

            // Simpan halaman sebelumnya
            pages.push(
                PdfPage::new(
                    Mm(210.0),
                    Mm(297.0),
                    ops,
                )
            );

            // Buat halaman baru
            ops = Vec::new();

            y = add_header(&mut ops);
        }

        // ======================================
        // DATA
        // ======================================

        add_text(
            &mut ops,
            18.0,
            y,
            format!("{}", id),
            8.0,
            false,
        );

        add_text(
            &mut ops,
            32.0,
            y,
            device_id.clone(),
            8.0,
            false,
        );

        add_text(
            &mut ops,
            68.0,
            y,
            timestamp.clone(),
            7.5,
            false,
        );

        add_text(
            &mut ops,
            125.0,
            y,
            classification.clone(),
            8.0,
            false,
        );

        add_text(
            &mut ops,
            172.0,
            y,
            format!(
                "{:.1}%",
                confidence * 100.0
            ),
            8.0,
            false,
        );

        // Garis pemisah setiap baris
        add_line(
            &mut ops,
            15.0,
            y - 4.0,
            195.0,
            y - 4.0,
        );

        // Jarak ke baris berikutnya
        y -= 12.0;
    }

    // ==========================================
    // SIMPAN HALAMAN TERAKHIR
    // ==========================================

    pages.push(
        PdfPage::new(
            Mm(210.0),
            Mm(297.0),
            ops,
        )
    );

    println!(
        "Jumlah halaman PDF: {}",
        pages.len()
    );

    // ==========================================
    // MASUKKAN SEMUA HALAMAN
    // ==========================================

    doc.with_pages(pages);

    // ==========================================
    // SIMPAN PDF
    // ==========================================

    let mut warnings = Vec::new();

    let pdf_bytes = doc.save(
        &PdfSaveOptions::default(),
        &mut warnings,
    );

    std::fs::write(
        "classification_report.pdf",
        &pdf_bytes,
    )?;

    println!(
        "✅ PDF berhasil dibuat!"
    );

    println!(
        "Ukuran PDF: {} bytes",
        pdf_bytes.len()
    );

    println!(
        "Jumlah warning: {}",
        warnings.len()
    );

    println!("================================");

    Ok(())
}