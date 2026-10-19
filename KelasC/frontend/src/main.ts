import './style.css'

interface Report {
  id: number
  device_id: string
  timestamp: string
  classification: string
  confidence: number
}

interface Stats {
  active_devices: number
  total_records: number
  average_confidence: number
}

async function getReports(): Promise<Report[]> {
  const response = await fetch('http://127.0.0.1:3000/reports')

  if (!response.ok) {
    throw new Error('Gagal mengambil data report')
  }

  return await response.json()
}

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="app-shell">

    <!-- SIDEBAR -->
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">IOT</div>
        <div>
          <strong>SMT 5</strong>
          <span>Monitoring System</span>
        </div>
      </div>

      <nav class="navigation">
        <a class="nav-item active" href="#">
          <span>⌂</span>
          Dashboard
        </a>

        <a class="nav-item" href="#">
          <span>◉</span>
          Monitoring
        </a>

        <a class="nav-item" href="#">
          <span>▤</span>
          Reports
        </a>

        <a class="nav-item" href="#">
          <span>↥</span>
          Firmware OTA
        </a>
      </nav>

      <div class="sidebar-bottom">
        <div class="system-info">
          <span class="system-dot"></span>
          <div>
            <strong>System Online</strong>
            <small>MQTT Connected</small>
          </div>
        </div>
      </div>
    </aside>


    <!-- MAIN CONTENT -->
    <main class="main-content">

      <!-- TOP BAR -->
      <header class="topbar">
        <div>
          <span class="page-label">CONTROL CENTER</span>
          <h1>ESP32-S3 Dashboard</h1>
          <p>Machine learning monitoring and firmware management</p>
        </div>

        <div class="connection-status">
          <span></span>
          MQTT CONNECTED
        </div>
      </header>


      <!-- MAIN RESULT -->
      <section class="hero-section">

        <div class="classification-panel">
          <div class="panel-header">
            <div>
              <span class="eyebrow">LATEST CLASSIFICATION</span>
              <h2>Current Result</h2>
            </div>

           <span class="result-tag" id="latest-tag">NORMAL</span>
          </div>

          <div class="classification-main">
            <div class="classification-value">
              NORMAL
            </div>

            <div class="confidence">
              <span>MODEL CONFIDENCE</span>
              <strong id="latest-confidence">0%</strong>
            </div>
          </div>

          <div class="result-meta">
            <div>
              <span>DEVICE</span>
              <strong id="latest-device">-</strong>
            </div>

            <div>
              <span>LAST UPDATE</span>
              <strong id="latest-update">-</strong>
            </div>

            <div>
              <span>STATUS</span>
              <strong class="blue-text">ONLINE</strong>
            </div>
          </div>
        </div>


        <!-- STAT COLUMN -->
        <div class="stats-column">

          <div class="stat-panel">
            <span>ACTIVE DEVICE</span>
            <strong id="active-devices">0</strong>
            <small>ESP32-S3 devices</small>
          </div>

          <div class="stat-panel">
            <span>TOTAL RECORDS</span>
            <strong id="total-records">0</strong>
            <small>Stored classifications</small>
          </div>

          <div class="stat-panel">
            <span>AVG. CONFIDENCE</span>
            <strong id="average-confidence">0%</strong>
            <small>Latest records</small>
          </div>

        </div>

      </section>


      <!-- HISTORY -->
      <section class="content-panel">

        <div class="panel-header history-header">
          <div>
            <span class="eyebrow">MONITORING DATA</span>
            <h2>Classification History</h2>
          </div>

          <div class="history-actions">
            <select>
              <option>All Devices</option>
              <option>ESP32-S3-001</option>
              <option>ESP32-S3-002</option>
            </select>

            <button class="report-button" id="generate-report-button">
              Generate Report
            </button>
          </div>
        </div>


        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>TIME</th>
                <th>DEVICE</th>
                <th>RESULT</th>
                <th>CONFIDENCE</th>
                <th>STATUS</th>
              </tr>
            </thead>

            <tbody id="report-table-body">

            </tbody>
          </table>
        </div>

      </section>


      <!-- OTA -->
      <section class="ota-panel">

        <div class="ota-heading">
          <div>
            <span class="eyebrow">DEVICE MANAGEMENT</span>
            <h2>Firmware OTA</h2>
            <p>Upload compiled firmware directly to ESP32-S3 devices.</p>
          </div>

          <div class="version-box">
            <span>CURRENT VERSION</span>
            <strong>v1.0.0</strong>
          </div>
        </div>


        <div class="ota-layout">

          <div class="upload-zone">
            <div class="upload-symbol">↑</div>

            <div>
              <strong>Select Firmware File</strong>
              <p>Compiled ESP32-S3 firmware (.bin)</p>
            </div>

            <button class="choose-button">
              Choose File
            </button>
          </div>


          <div class="firmware-details">

            <div>
              <span>SELECTED FILE</span>
              <strong>firmware_v1.1.0.bin</strong>
            </div>

            <div>
              <span>FILE SIZE</span>
              <strong>842 KB</strong>
            </div>

            <div>
              <span>TARGET</span>
              <strong>ESP32-S3</strong>
            </div>

          </div>

        </div>

        <button class="ota-button">
          START OTA UPDATE
        </button>

      </section>


      <footer>
        IoT SMT 5 · ESP32-S3 Monitoring System
      </footer>

    </main>

  </div>
`
function displayReports(reports: Report[]) {
  const tableBody =
    document.querySelector<HTMLTableSectionElement>(
      '#report-table-body'
    )

  if (!tableBody) return

  tableBody.innerHTML = reports
    .map((report) => {
      const time = report.timestamp.split(' ')[1] ?? '-'
      const confidence =
        (report.confidence * 100).toFixed(1)

      const classification =
        report.classification.toUpperCase()

      const resultClass =
        classification === 'AMAN' ||
        classification === 'NORMAL'
          ? 'normal'
          : 'warning'

      return `
        <tr>
          <td>${time}</td>

          <td>${report.device_id}</td>

          <td>
            <span class="result ${resultClass}">
              ${classification}
            </span>
          </td>

          <td>${confidence}%</td>

          <td>
            <span class="status-online">
              ONLINE
            </span>
          </td>
        </tr>
      `
    })
    .join('')
}

getReports()
  .then((reports) => {
    console.log('Data dari Rust:', reports)

    displayReports(reports)
    displayLatestReport(reports)
  })
  .catch((error) => {
    console.error('Error:', error)
  })

  async function getStats(): Promise<Stats> {
  const response = await fetch(
    'http://127.0.0.1:3000/stats'
  )

  if (!response.ok) {
    throw new Error('Gagal mengambil statistik')
  }

  return await response.json()
}

function displayStats(stats: Stats) {
  const activeDevices =
    document.querySelector('#active-devices')

  const totalRecords =
    document.querySelector('#total-records')

  const averageConfidence =
    document.querySelector('#average-confidence')

  if (activeDevices) {
    activeDevices.textContent =
      stats.active_devices.toString()
  }

  if (totalRecords) {
    totalRecords.textContent =
      stats.total_records.toString()
  }

  if (averageConfidence) {
    averageConfidence.textContent =
      `${(stats.average_confidence * 100).toFixed(1)}%`
  }
}

getStats()
  .then((stats) => {
    console.log('Stats dari Rust:', stats)

    displayStats(stats)
  })
  .catch((error) => {
    console.error('Stats error:', error)
  })

  function displayLatestReport(reports: Report[]) {
  if (reports.length === 0) {
    return
  }

  const latest = reports[0]

  const classification =
    latest.classification.toUpperCase()

  const confidence =
    (latest.confidence * 100).toFixed(1)

  const latestTag =
    document.querySelector('#latest-tag')

  const latestClassification =
    document.querySelector('#latest-classification')

  const latestConfidence =
    document.querySelector('#latest-confidence')

  const latestDevice =
    document.querySelector('#latest-device')

  const latestUpdate =
    document.querySelector('#latest-update')

  if (latestTag) {
    latestTag.textContent = classification
  }

  if (latestClassification) {
    latestClassification.textContent =
      classification
  }

  if (latestConfidence) {
    latestConfidence.textContent =
      `${confidence}%`
  }

  if (latestDevice) {
    latestDevice.textContent =
      latest.device_id
  }

  if (latestUpdate) {
    latestUpdate.textContent =
      latest.timestamp
  }
}

// ==========================================
// GENERATE PDF REPORT
// ==========================================

const generateReportButton =
  document.querySelector<HTMLButtonElement>(
    '#generate-report-button'
  )

generateReportButton?.addEventListener(
  'click',
  async () => {

    try {
      // Ambil PDF dari Rust Backend
      const response = await fetch(
        'http://127.0.0.1:3000/reports/pdf'
      )

      if (!response.ok) {
        throw new Error(
          'Gagal membuat PDF report'
        )
      }

      // Ubah response menjadi file/blob
      const blob = await response.blob()

      // Buat URL sementara
      const url =
        URL.createObjectURL(blob)

      // Buat link download
      const link =
        document.createElement('a')

      link.href = url
      link.download =
        'classification_report.pdf'

      // Jalankan download
      document.body.appendChild(link)

      link.click()

      // Bersihkan
      document.body.removeChild(link)

      URL.revokeObjectURL(url)

      console.log(
        'PDF berhasil didownload'
      )

    } catch (error) {

      console.error(
        'Gagal download PDF:',
        error
      )

      alert(
        'Gagal membuat report PDF'
      )
    }
  }
)