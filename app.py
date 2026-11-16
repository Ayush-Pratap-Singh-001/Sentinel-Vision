import os
import time
import uuid
import base64
import shutil
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="Sentinel-Vision Compliance Monitor")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL_PATH = BASE_DIR / "sentinel_ppe_best.pt"
if not MODEL_PATH.exists():
    print(f"WARNING: {MODEL_PATH} not found! Place sentinel_ppe_best.pt in {BASE_DIR}")

model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else None

# Global state for active video processing & analytics
class MonitorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.video_source: Optional[str] = None
        self.is_running: bool = False
        self.conf_threshold: float = 0.40
        # Normalized hazardous zone polygon [(x, y), ...] in 0..1 coordinates
        self.hazard_zone: List[List[float]] = [
            [0.62, 0.45],
            [0.95, 0.45],
            [0.95, 0.95],
            [0.62, 0.95],
        ]
        self.zone_enabled: bool = True

        # Live Analytics
        self.fps: float = 0.0
        self.frame_index: int = 0
        self.video_timestamp_sec: float = 0.0
        self.active_workers: int = 0
        self.compliant_workers: int = 0
        self.non_compliant_workers: int = 0
        self.zone_intrusions: int = 0
        self.compliance_rate: float = 100.0
        self.risk_score: int = 0
        self.ppe_counts: Dict[str, int] = defaultdict(int)
        self.compliance_history: deque = deque(maxlen=60)
        self.incidents: List[dict] = []

        # Temporal validation trackers: track_id -> consecutive violation frames
        self.violation_streak: Dict[int, int] = defaultdict(int)
        self.logged_incidents: set = set()  # (track_id, violation_type)

    def reset_metrics(self):
        with self.lock:
            self.frame_index = 0
            self.video_timestamp_sec = 0.0
            self.active_workers = 0
            self.compliant_workers = 0
            self.non_compliant_workers = 0
            self.zone_intrusions = 0
            self.compliance_rate = 100.0
            self.risk_score = 0
            self.ppe_counts = defaultdict(int)
            self.compliance_history.clear()
            self.incidents.clear()
            self.violation_streak.clear()
            self.logged_incidents.clear()

state = MonitorState()

class ConfigUpdate(BaseModel):
    conf_threshold: Optional[float] = None
    zone_enabled: Optional[bool] = None
    hazard_zone: Optional[List[List[float]]] = None


def box_iou_or_inside(worker_box, item_box) -> bool:
    """Checks if a PPE item's center lies inside or overlaps strongly with a worker's bounding box."""
    wx1, wy1, wx2, wy2 = worker_box
    ix1, iy1, ix2, iy2 = item_box
    icx = (ix1 + ix2) / 2.0
    icy = (iy1 + iy2) / 2.0
    # Expand worker box slightly (10%) to catch helmets/boots near edges
    pad_x = (wx2 - wx1) * 0.10
    pad_y = (wy2 - wy1) * 0.10
    return (wx1 - pad_x <= icx <= wx2 + pad_x) and (wy1 - pad_y <= icy <= wy2 + pad_y)


def encode_snapshot(frame: np.ndarray, box) -> str:
    """Crops or annotates incident snapshot and returns base64 JPEG."""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = 30
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = frame
    _, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def generate_frames():
    """Generator yielding MJPEG frames with real-time YOLO tracking, PPE association & zone alerts."""
    if model is None:
        return

    cap = cv2.VideoCapture(state.video_source)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0 or video_fps > 120:
        video_fps = 25.0

    state.is_running = True
    prev_time = time.time()

    while state.is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video automatically for continuous dashboard monitoring
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            state.violation_streak.clear()
            continue

        h, w, _ = frame.shape
        state.frame_index += 1
        state.video_timestamp_sec = round(state.frame_index / video_fps, 1)

        # Run YOLO11 with ByteTrack tracking
        results = model.track(
            frame,
            persist=True,
            conf=state.conf_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        names = results.names
        workers = []
        ppe_items = []
        frame_ppe_counts = defaultdict(int)

        if results.boxes is not None and len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            track_ids = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else [-1] * len(boxes_xyxy)
            )

            for box, cls_id, conf, tid in zip(boxes_xyxy, classes, confs, track_ids):
                label = names[cls_id].lower()
                frame_ppe_counts[label] += 1
                if label == "person":
                    workers.append({"id": int(tid), "box": box, "conf": float(conf)})
                else:
                    ppe_items.append({"label": label, "box": box, "conf": float(conf)})

        # Build Hazardous Zone Polygon in pixel coordinates
        zone_poly = None
        if state.zone_enabled and len(state.hazard_zone) >= 3:
            zone_poly = np.array(
                [[int(pt[0] * w), int(pt[1] * h)] for pt in state.hazard_zone],
                dtype=np.int32,
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_poly], (0, 140, 255))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [zone_poly], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(
                frame,
                "RESTRICTED HAZARD ZONE",
                (zone_poly[0][0] + 8, max(25, zone_poly[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )

        # Draw non-person PPE detections lightly
        for item in ppe_items:
            ix1, iy1, ix2, iy2 = [int(v) for v in item["box"]]
            is_neg = item["label"].startswith("no_")
            color = (60, 60, 235) if is_neg else (235, 180, 50)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 1)
            cv2.putText(
                frame,
                f"{item['label']} {item['conf']:.2f}",
                (ix1, max(15, iy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )

        # Evaluate each tracked worker for PPE compliance & Zone Intrusion
        compliant_cnt = 0
        non_compliant_cnt = 0
        intrusion_cnt = 0

        for worker in workers:
            wx1, wy1, wx2, wy2 = [int(v) for v in worker["box"]]
            tid = worker["id"]
            foot_point = (int((wx1 + wx2) / 2), wy2)

            associated_positive = []
            associated_violations = []

            for item in ppe_items:
                if box_iou_or_inside(worker["box"], item["box"]):
                    lbl = item["label"]
                    if lbl.startswith("no_"):
                        associated_violations.append(lbl.replace("no_", "Missing ").title())
                    else:
                        associated_positive.append(lbl)

            # Check explicit missing mandatory PPE if neither helmet nor no_helmet detected
            has_helmet = "helmet" in associated_positive
            has_vest = "vest" in associated_positive
            if not has_helmet and "Missing Helmet" not in associated_violations:
                associated_violations.append("Missing Helmet")
            if not has_vest and "Missing Vest" not in associated_violations:
                associated_violations.append("Missing Vest")

            # Check Hazardous Zone Intrusion (foot center inside polygon)
            in_hazard_zone = False
            if zone_poly is not None:
                inside = cv2.pointPolygonTest(zone_poly, foot_point, False)
                if inside >= 0:
                    in_hazard_zone = True
                    intrusion_cnt += 1
                    associated_violations.append("Hazard Zone Intrusion")

            is_compliant = len(associated_violations) == 0

            if is_compliant:
                compliant_cnt += 1
                state.violation_streak[tid] = 0
                box_color = (46, 204, 113)  # Green
                status_text = f"Worker #{tid} | COMPLIANT"
            else:
                non_compliant_cnt += 1
                state.violation_streak[tid] += 1
                box_color = (0, 165, 255) if (in_hazard_zone and len(associated_violations) == 1) else (59, 59, 239)
                status_text = f"Worker #{tid} | {', '.join(associated_violations)}"

                # Temporal Validation: Log incident only after 8 consecutive frames of violation
                if state.violation_streak[tid] == 8:
                    incident_key = (tid, tuple(sorted(associated_violations)))
                    if incident_key not in state.logged_incidents:
                        state.logged_incidents.add(incident_key)
                        mins = int(state.video_timestamp_sec // 60)
                        secs = int(state.video_timestamp_sec % 60)
                        snapshot_b64 = encode_snapshot(frame, worker["box"])
                        with state.lock:
                            state.incidents.insert(
                                0,
                                {
                                    "id": str(uuid.uuid4())[:8],
                                    "timestamp": f"{mins:02d}:{secs:02d}",
                                    "worker_id": f"Worker #{tid}" if tid != -1 else "Unassigned",
                                    "violations": associated_violations,
                                    "severity": "CRITICAL" if in_hazard_zone or "Missing Helmet" in associated_violations else "HIGH",
                                    "snapshot": snapshot_b64,
                                },
                            )
                            # Keep latest 30 incidents
                            state.incidents = state.incidents[:30]

            # Draw Worker Bounding Box & Status Pill
            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), box_color, 2)
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (wx1, max(0, wy1 - th - 10)), (wx1 + tw + 8, wy1), box_color, -1)
            cv2.putText(
                frame,
                status_text,
                (wx1 + 4, max(14, wy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # Compute instantaneous FPS & Risk Score
        now = time.time()
        inst_fps = 1.0 / max(1e-5, (now - prev_time))
        prev_time = now

        total_w = len(workers)
        comp_rate = round((compliant_cnt / total_w) * 100.0, 1) if total_w > 0 else 100.0
        raw_risk = (non_compliant_cnt * 22) + (intrusion_cnt * 35)
        risk_score = min(100, raw_risk)

        with state.lock:
            state.fps = round(0.85 * state.fps + 0.15 * inst_fps, 1) if state.fps > 0 else round(inst_fps, 1)
            state.active_workers = total_w
            state.compliant_workers = compliant_cnt
            state.non_compliant_workers = non_compliant_cnt
            state.zone_intrusions = intrusion_cnt
            state.compliance_rate = comp_rate
            state.risk_score = risk_score
            state.ppe_counts = dict(frame_ppe_counts)
            if state.frame_index % 10 == 0:
                state.compliance_history.append(
                    {
                        "time": f"{state.video_timestamp_sec}s",
                        "compliance": comp_rate,
                        "risk": risk_score,
                    }
                )

        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )

    cap.release()


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found in static/</h1>", status_code=404)
    return index_file.read_text(encoding="utf-8")


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format. Upload MP4, AVI, MOV, or MKV.")

    # Stop current stream before replacing source
    state.is_running = False
    time.sleep(0.2)

    save_path = UPLOAD_DIR / f"active_video{ext}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    state.video_source = str(save_path)
    state.reset_metrics()
    return {"status": "ok", "filename": file.filename}


@app.get("/api/video_feed")
def video_feed():
    if not state.video_source or not Path(state.video_source).exists():
        raise HTTPException(status_code=404, detail="No video uploaded yet.")
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/analytics")
def get_analytics():
    with state.lock:
        return JSONResponse(
            {
                "is_running": state.is_running,
                "has_video": state.video_source is not None,
                "fps": state.fps,
                "timestamp_sec": state.video_timestamp_sec,
                "active_workers": state.active_workers,
                "compliant_workers": state.compliant_workers,
                "non_compliant_workers": state.non_compliant_workers,
                "zone_intrusions": state.zone_intrusions,
                "compliance_rate": state.compliance_rate,
                "risk_score": state.risk_score,
                "ppe_counts": state.ppe_counts,
                "history": list(state.compliance_history),
                "incidents": state.incidents,
                "config": {
                    "conf_threshold": state.conf_threshold,
                    "zone_enabled": state.zone_enabled,
                    "hazard_zone": state.hazard_zone,
                },
            }
        )


@app.post("/api/config")
def update_config(cfg: ConfigUpdate):
    if cfg.conf_threshold is not None:
        state.conf_threshold = max(0.1, min(0.95, cfg.conf_threshold))
    if cfg.zone_enabled is not None:
        state.zone_enabled = cfg.zone_enabled
    if cfg.hazard_zone is not None and len(cfg.hazard_zone) >= 3:
        state.hazard_zone = cfg.hazard_zone
    return {"status": "updated"}