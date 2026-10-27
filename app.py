from __future__ import annotations

# Crucial Windows fix: Import torch first to initialize C++ runtime / c10.dll before pandas/pyarrow/streamlit
import torch

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, List

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

_last_beep_time = 0.0

def trigger_audio_alarm():
    """Trigger non-blocking Windows hardware siren/beep for abandoned object alert."""
    global _last_beep_time
    now = time.time()
    if now - _last_beep_time < 1.5:
        return
    _last_beep_time = now

    def _play():
        if _HAS_WINSOUND:
            try:
                for _ in range(2):
                    winsound.Beep(2200, 160)
                    time.sleep(0.04)
            except Exception:
                pass

    threading.Thread(target=_play, daemon=True).start()

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.inference.pipeline import SurveillancePipeline
from src.temporal.states import AbandonmentState
from scripts.create_local_dataset import create_all_local_scenarios


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Unattended Object Detection – Edge AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.85; transform: scale(1.005); }
        100% { opacity: 1; transform: scale(1); }
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .alert-box {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 14px 20px;
        border-radius: 6px;
        margin-bottom: 15px;
        font-weight: 600;
        color: #991B1B;
        font-size: 1.1rem;
        animation: pulse 1s infinite;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_sample_videos_exist():
    """Ensure local synthetic test videos are present for instant testing."""
    sample_dir = PROJECT_ROOT / "data" / "local" / "videos"
    if not (sample_dir / "campus_true_abandoned.mp4").exists():
        create_all_local_scenarios(PROJECT_ROOT / "data" / "local")


ensure_sample_videos_exist()

# Load default config
config_path = PROJECT_ROOT / "configs" / "system.yaml"
base_config = load_yaml(config_path)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/cctv.png", width=64)
st.sidebar.title("System Controls")

st.sidebar.subheader("1. Speed & Detector Settings")
model_options = ["yolov8n.pt"]
best_pt = PROJECT_ROOT / "models" / "weights" / "best.pt"
if best_pt.exists():
    model_options.insert(0, str(best_pt))

selected_model = st.sidebar.selectbox("Model Weights", model_options, index=0)

perf_mode = st.sidebar.select_slider(
    "⚡ Performance / FPS Target",
    options=["🚀 Ultra Fast (384px - 30+ FPS)", "⚖️ Balanced (480px - 20-30 FPS)", "🎯 Max Detail (640px)"],
    value="🚀 Ultra Fast (384px - 30+ FPS)",
    help="Ultra Fast downsizes inference resolution for high frame rates (>30 FPS) on CPU.",
)
imgsz_map = {
    "🚀 Ultra Fast (384px - 30+ FPS)": 384,
    "⚖️ Balanced (480px - 20-30 FPS)": 480,
    "🎯 Max Detail (640px)": 640,
}
selected_imgsz = imgsz_map[perf_mode]

conf_thresh = st.sidebar.slider(
    "Detection Sensitivity (Confidence)",
    min_value=0.05,
    max_value=0.80,
    value=0.15,
    step=0.05,
    help="Lower values (0.10 - 0.20) allow bags and backpacks to be detected easily in challenging lighting."
)
iou_thresh = st.sidebar.slider("IoU NMS Threshold", 0.10, 0.90, float(base_config["detection"]["iou_threshold"]), 0.05)

st.sidebar.subheader("2. Temporal & Association")
abandon_duration = st.sidebar.slider(
    "Abandonment Duration (s)", 3.0, 60.0, float(base_config["temporal"]["abandonment_time_threshold"]), 1.0
)
owner_distance = st.sidebar.slider(
    "Owner Separation Dist (px)", 50.0, 500.0, float(base_config["temporal"]["owner_distance_threshold"]), 10.0
)
movement_thresh = st.sidebar.slider(
    "Stationary Displacement Thresh (px)", 5.0, 100.0, float(base_config["temporal"]["object_movement_threshold"]), 5.0
)

st.sidebar.subheader("3. Overlays & Audible Alarm")
enable_audio_alarm = st.sidebar.checkbox("🔊 Audible Siren Alarm on Abandonment", True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_ids = st.sidebar.checkbox("Show Track IDs", True)
show_assoc = st.sidebar.checkbox("Show Owner Link & Distance", True)
show_timer = st.sidebar.checkbox("Show Abandonment Timer", True)
show_fps = st.sidebar.checkbox("Show Live FPS & Latency", True)

# Build runtime config
run_config = base_config.copy()
run_config["detection"]["model_path"] = selected_model
run_config["detection"]["confidence_threshold"] = conf_thresh
run_config["detection"]["iou_threshold"] = iou_thresh
run_config["detection"]["imgsz"] = selected_imgsz
run_config["tracking"]["track_thresh"] = min(conf_thresh, 0.20)
run_config["temporal"]["abandonment_time_threshold"] = abandon_duration
run_config["temporal"]["owner_distance_threshold"] = owner_distance
run_config["temporal"]["object_movement_threshold"] = movement_thresh
run_config["visualization"]["show_bounding_boxes"] = show_boxes
run_config["visualization"]["show_track_ids"] = show_ids
run_config["visualization"]["show_owner_association"] = show_assoc
run_config["visualization"]["show_timer"] = show_timer
run_config["visualization"]["show_fps"] = show_fps

# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-title">🚨 Unattended Object Detection – Edge AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Spatial Multi-Object Tracking + Temporal Ownership Association + Deterministic Reasoning Machine</div>',
    unsafe_allow_html=True,
)

# Input Source Selector
tab_sample, tab_upload, tab_webcam = st.tabs(["📁 Sample Staged Scenarios", "⬆️ Upload Video", "📹 Live Camera"])

video_source = None
sample_vids_dir = PROJECT_ROOT / "data" / "local" / "videos"

with tab_sample:
    sample_choice = st.selectbox(
        "Select Staged Campus Scenario:",
        [
            ("campus_true_abandoned.mp4", "Scenario 1: True Abandoned Luggage (Owner leaves, bag sits stationary -> Alert)"),
            ("campus_owner_returns.mp4", "Scenario 2: Owner Returns (Owner departs, returns before threshold -> No Alert)"),
            ("campus_normal_attended.mp4", "Scenario 3: Attended Luggage (Owner remains standing beside bag -> No Alert)"),
        ],
        format_func=lambda x: f"{x[1]}",
    )
    if sample_choice:
        video_source = str(sample_vids_dir / sample_choice[0])

with tab_upload:
    uploaded_file = st.file_uploader("Upload surveillance MP4 / AVI video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_source = tfile.name

with tab_webcam:
    cam_type = st.radio(
        "Camera Connection Method:",
        ["Device Index (Laptop Cam or Virtual Phone Cam via Iriun/DroidCam)", "WiFi Phone Stream (IP Webcam URL)"],
        horizontal=True,
    )
    if cam_type.startswith("Device Index"):
        device_idx = st.selectbox(
            "Select Camera Device Index:",
            [
                (0, "Device 0 (Default Laptop Camera)"),
                (1, "Device 1 (Phone Camera via Iriun / DroidCam / External USB)"),
                (2, "Device 2 (Secondary Camera)"),
            ],
            format_func=lambda x: x[1],
            index=0,
        )
        webcam_enabled = st.checkbox("Enable Live Camera Feed", key="chk_device_cam")
        if webcam_enabled:
            video_source = device_idx[0]
    else:
        st.info("📱 **Tip:** Install the free **'IP Webcam'** app on your phone, tap **'Start server'**, and paste the URL shown on your phone screen below (e.g., `http://192.168.1.15:8080/video`).")
        ip_url = st.text_input("Enter Phone IP Stream URL:", placeholder="http://192.168.1.X:8080/video")
        stream_enabled = st.checkbox("Connect to Phone Stream", key="chk_ip_cam")
        if stream_enabled and ip_url.strip():
            video_source = ip_url.strip()

# Action Controls
col_btn1, col_btn2, _ = st.columns([1, 1, 4])
start_processing = col_btn1.button("▶️ Start Processing", type="primary", use_container_width=True)
stop_processing = col_btn2.button("⏹️ Stop Processing", use_container_width=True)

# Layout: Video on Left, Live Telemetry & Log on Right
col_left, col_right = st.columns([3, 2])

with col_left:
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col_right:
    st.subheader("📊 Live Telemetry")
    metric_cols = st.columns(4)
    metric_fps = metric_cols[0].metric("FPS", "0.0")
    metric_lat = metric_cols[1].metric("Latency", "0.0 ms")
    metric_persons = metric_cols[2].metric("Persons", "0")
    metric_bags = metric_cols[3].metric("Bags", "0")

    hw_cols = st.columns(3)
    metric_cpu = hw_cols[0].metric("CPU", "0%")
    metric_gpu = hw_cols[1].metric("GPU Load", "N/A")
    metric_power = hw_cols[2].metric("Power", "N/A")

    st.subheader("📜 Temporal Event & Alert Log")
    log_placeholder = st.empty()

# Persistent session state for event logs
if "event_logs" not in st.session_state or stop_processing:
    st.session_state.event_logs = []

if start_processing and video_source is not None:
    pipeline = SurveillancePipeline(run_config)
    cap = cv2.VideoCapture(video_source)
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    fps_nominal = float(fps_val) if (fps_val and fps_val > 0 and not np.isnan(fps_val)) else 30.0

    frame_idx = 0
    active_alerts_banner = []
    consecutive_misses = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_misses += 1
                if consecutive_misses > 30:
                    break
                time.sleep(0.015)
                continue
            consecutive_misses = 0

            # Downscale high-resolution webcam/phone frames to max width 640
            # This accelerates OpenCV processing and eliminates browser WebSocket transmission bottlenecks
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)

            timestamp = frame_idx / fps_nominal
            res = pipeline.process_frame(frame, timestamp, frame_index=frame_idx, render_annotated=True)

            # Check for new alerts
            if res.alerts:
                for a in res.alerts:
                    st.session_state.event_logs.insert(0, {
                        "Time": f"{a.timestamp:.1f}s",
                        "Object": f"B{a.bag_id:02d}",
                        "Owner": f"P{a.owner_id or 0:02d}",
                        "State": a.state.value,
                        "Duration": f"{a.separation_duration:.1f}s",
                        "Alert": "🚨 ABANDONED OBJECT DETECTED",
                    })

            # Check active states for transitions
            for bid, st_info in res.bag_states.items():
                if st_info.current_state == AbandonmentState.ABANDONED:
                    active_alerts_banner.append(f"Bag B{bid:02d} left unattended for {st_info.separation_duration:.1f}s!")

            # Update Video Frame
            rgb_frame = cv2.cvtColor(res.annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update Alert Banner & Trigger Audible Siren
            if active_alerts_banner:
                if enable_audio_alarm:
                    trigger_audio_alarm()
                unique_alerts = list(set(active_alerts_banner))
                # Web Audio API inline script to play audible chime directly in browser as well
                audio_tag = """
                <script>
                (function() {
                    try {
                        var AudioCtx = window.AudioContext || window.webkitAudioContext;
                        if (AudioCtx) {
                            var ctx = new AudioCtx();
                            var osc = ctx.createOscillator();
                            var gain = ctx.createGain();
                            osc.type = 'sawtooth';
                            osc.frequency.setValueAtTime(880, ctx.currentTime);
                            osc.frequency.linearRampToValueAtTime(1320, ctx.currentTime + 0.15);
                            gain.gain.setValueAtTime(0.2, ctx.currentTime);
                            gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.2);
                            osc.connect(gain);
                            gain.connect(ctx.destination);
                            osc.start();
                            osc.stop(ctx.currentTime + 0.2);
                        }
                    } catch(e){}
                })();
                </script>
                """
                alert_placeholder.markdown(
                    f'<div class="alert-box">🚨 <b>SECURITY ALERT:</b> {", ".join(unique_alerts)}</div>' + audio_tag,
                    unsafe_allow_html=True,
                )
            else:
                alert_placeholder.empty()

            # Update telemetry metrics every 3 frames to maintain high FPS without WebSocket choking
            if frame_idx % 3 == 0:
                metric_fps.metric("FPS", f"{res.fps:.1f}")
                metric_lat.metric("Latency", f"{res.latency_ms:.1f} ms")
                metric_persons.metric("Persons", f"{len(res.person_tracks)}")
                metric_bags.metric("Bags", f"{len(res.bag_tracks)}")

                hw = res.hardware_metrics
                metric_cpu.metric("CPU", f"{hw.get('cpu_util_percent', 0)}%")
                gpu_u = hw.get("gpu_util_percent")
                metric_gpu.metric("GPU Load", f"{gpu_u}%" if gpu_u is not None else "N/A")
                pwr = hw.get("power_w")
                metric_power.metric("Power", f"{pwr} W" if pwr is not None else "N/A")

            # Update Event Table safely (bulletproof against pyarrow or rendering errors)
            if st.session_state.event_logs and frame_idx % 3 == 0:
                df_log = pd.DataFrame(st.session_state.event_logs[:15])
                try:
                    log_placeholder.dataframe(df_log, use_container_width=True, hide_index=True)
                except Exception:
                    log_placeholder.markdown(
                        df_log.to_html(index=False, classes="table table-bordered table-sm"),
                        unsafe_allow_html=True
                    )

            frame_idx += 1
            active_alerts_banner.clear()

    finally:
        cap.release()
