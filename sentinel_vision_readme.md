# SENTINEL-VISION: AI-Powered Worker Safety & PPE Compliance Monitor

**Sentinel-Vision** is a real-time computer-vision safety monitoring system built for construction sites, underground mines, factories, and other hazardous industrial environments. Developed for **Smart India Hackathon (SIH) 2026 — SIH26039** (*AI-Powered Underground Mine Safety, Monitoring and Rescue System*), the system detects workers and Personal Protective Equipment (PPE), tracks individuals across frames, monitors configurable virtual hazard zones, reduces false alerts through temporal validation, and delivers an interactive real-time compliance analytics dashboard.

---

## Key Features & Project Outcomes

* **Custom-Trained PPE Detection (`YOLO11s`):** Fine-tuned on the `construction-ppe` dataset to detect workers (`person`) alongside compliant gear (`helmet`, `vest`, `gloves`, `boots`, `goggles`) and non-compliant states (`no_helmet`, `no_vest`, `no_gloves`, `no_boots`, `no_goggle`).
* **Multi-Object Worker Tracking & Spatial PPE Association:** Integrates **ByteTrack** to assign persistent IDs (`Worker #1`, `Worker #2`, etc.) and spatially associates detected PPE bounding boxes with individual workers.
* **Configurable Virtual Hazardous Zones:** Evaluates worker foot-point geometry against restricted danger-zone polygons in real time to flag unauthorized perimeter intrusions.
* **Temporal Alert Debouncing & Risk Scoring:** Eliminates single-frame false positives by requiring violations to persist across **8 consecutive frames** before logging a verified incident, calculating a live **Site Risk Score (0–100)**, and capturing timestamped visual snapshot evidence.
* **Interactive Web Compliance Dashboard:** Streams live annotated MJPEG video alongside real-time KPI cards, temporal compliance/risk trend charts, model benchmark comparisons, and an incident evidence feed.

---

## Sustainable Development Goals (SDGs) Alignment

| SDG | Goal | Project Impact |
| :--- | :--- | :--- |
| **SDG 3** | Good Health and Well-Being | Prevents workplace injuries and fatalities by detecting unsafe conditions and missing PPE early. |
| **SDG 8** | Decent Work and Economic Growth | Promotes safer industrial and mining environments through continuous automated monitoring. |
| **SDG 9** | Industry, Innovation and Infrastructure | Applies edge-ready computer vision to modernize industrial safety infrastructure. |

---

## Project Directory Structure

```text
sentinel_vision_app/
├── sentinel_ppe_best.pt          # Custom-trained YOLO11s model weights
├── model_comparison_metrics.csv  # Benchmark results (Stock vs. Custom YOLO11s)
├── app.py                        # FastAPI backend, tracking engine & analytics API
├── requirements.txt              # Python package dependencies
├── README.md                     # Project documentation
├── static/
│   └── index.html                # Real-time compliance monitor web dashboard
└── uploads/                      # Temporary storage for uploaded site/mine videos
```

---

## Installation & Setup

### 1. Prerequisites
* Python **3.10+**
* Trained model weights (`sentinel_ppe_best.pt`) placed in the root `sentinel_vision_app/` directory.

### 2. Install Dependencies
Navigate to the project folder and install the required Python packages:

```bash
cd sentinel_vision_app
pip install -r requirements.txt
```

### 3. Run the Compliance Monitor Server
Start the FastAPI backend with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Launch the Dashboard
Open your browser and navigate to:
```text
http://localhost:8000
```

---

## Usage Guide

1. **Upload Video Feed:** Click **Upload Site / Mine Video** in the top navigation bar and select an `.mp4`, `.avi`, `.mov`, or `.mkv` recording.
2. **Monitor Real-Time Analytics:** View live FPS, active worker headcount, PPE compliance rate (%), active violations, hazard zone intrusions, and the overall Site Risk Score.
3. **Adjust Detection Controls:** Use the confidence threshold slider (`0.20` – `0.80`) or toggle the **Virtual Hazard Zone** overlay directly from the video header bar.
4. **Inspect Incident Evidence:** Whenever a worker violates PPE rules or enters a restricted zone for 8+ consecutive frames, a timestamped snapshot and severity badge are automatically logged in the right-hand **Timestamped Incident Evidence** panel.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Sentinel-Vision compliance dashboard (`static/index.html`). |
| `POST` | `/api/upload_video` | Uploads a workplace/mine video file and resets session analytics. |
| `GET` | `/api/video_feed` | Streams real-time MJPEG video annotated with tracking boxes, PPE labels, and hazard zones. |
| `GET` | `/api/analytics` | Returns live JSON telemetry (KPIs, class counts, temporal history, and verified incident snapshots). |
| `POST` | `/api/config` | Dynamically updates detection confidence threshold, zone toggle state, or polygon coordinates. |

---

## Model Training & Benchmarking Summary

* **Base Architecture:** `YOLO11s` (Ultralytics)
* **Dataset:** `construction-ppe` (Train/Val split, $640 \times 640$ resolution)
* **Benchmarking:** Evaluated directly against stock COCO-pretrained `yolo11s.pt` on the validation set (`model_comparison_metrics.csv`), demonstrating full multi-class PPE recognition where stock COCO weights only detect generic `person` instances.