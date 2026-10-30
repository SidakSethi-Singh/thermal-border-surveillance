"""
live_camera.py
---------------
Live camera tracking demo for the hackathon project.

Streams your PHONE's camera into the browser (via WebRTC), runs YOLOv8
object detection on your LAPTOP (GPU if available), and for every
detected PERSON adds a threat / normal / safe tag based on movement —
same rule-based logic style as track_analyze.py, just live instead of
on pre-recorded video.

Run:
    streamlit run live_camera.py

Then open the printed URL. See the setup notes at the bottom of this
file for how to connect your phone (HTTPS / ngrok requirement).
"""

import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# General-purpose pretrained model (COCO: person, chair, dining table, etc.)
# Swap to your thermal "best.pt" if you want FLIR-only classes instead.
MODEL_PATH = "yolov8n.pt"

INFER_EVERY_N_FRAMES = 2        # skip frames to keep latency low
RESIZE_WIDTH = 640              # downscale before inference
CONF_THRESHOLD = 0.4

# Movement-classification thresholds (tune these against your own footage)
SPEED_THREAT_PX_PER_FRAME = 28     # fast motion -> threat
SPEED_NORMAL_PX_PER_FRAME = 6      # slow/stationary -> safe, else normal
CROUCH_ASPECT_RATIO = 0.9          # bbox height/width below this -> crouching
GROUP_DISTANCE_PX = 120            # persons closer than this -> "group" bump
TRACK_HISTORY_LEN = 8
TRACK_MATCH_MAX_DIST = 80          # max px to match a detection to an existing track
TRACK_STALE_SECONDS = 1.5          # drop a track if unseen this long

COCO_PERSON_CLASS_ID = 0

COLORS = {
    "threat": (0, 0, 255),      # red   (BGR)
    "normal": (0, 210, 255),    # amber
    "safe": (0, 200, 0),        # green
    "object": (255, 200, 0),    # cyan-ish for non-person objects
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------------------------------------------------------------
# Simple centroid tracker (no extra deps) just to get per-person movement
# ---------------------------------------------------------------------------


class PersonTrack:
    __slots__ = ("id", "positions", "last_seen", "last_bbox")

    def __init__(self, track_id, cx, cy, bbox, t):
        self.id = track_id
        self.positions = deque(maxlen=TRACK_HISTORY_LEN)
        self.positions.append((cx, cy, t))
        self.last_seen = t
        self.last_bbox = bbox

    def update(self, cx, cy, bbox, t):
        self.positions.append((cx, cy, t))
        self.last_seen = t
        self.last_bbox = bbox

    def speed_px_per_frame(self):
        if len(self.positions) < 2:
            return 0.0
        (x0, y0, t0), (x1, y1, t1) = self.positions[-2], self.positions[-1]
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return dist


class CentroidTracker:
    def __init__(self):
        self.tracks = {}
        self._next_id = 1

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections, t):
        """detections: list of (cx, cy, bbox). Returns list of (track_id, bbox, speed, aspect)."""
        # drop stale tracks
        stale = [tid for tid, tr in self.tracks.items() if t - tr.last_seen > TRACK_STALE_SECONDS]
        for tid in stale:
            del self.tracks[tid]

        unmatched = list(range(len(detections)))
        results = []

        for tid, tr in list(self.tracks.items()):
            if not unmatched:
                break
            last_cx, last_cy, _ = tr.positions[-1]
            best_j, best_dist = None, TRACK_MATCH_MAX_DIST
            for j in unmatched:
                cx, cy, _ = detections[j]
                dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None:
                cx, cy, bbox = detections[best_j]
                tr.update(cx, cy, bbox, t)
                unmatched.remove(best_j)

        for j in unmatched:
            cx, cy, bbox = detections[j]
            tid = self._new_id()
            self.tracks[tid] = PersonTrack(tid, cx, cy, bbox, t)

        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = tr.last_bbox
            w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            aspect = h / w
            results.append((tid, tr.last_bbox, tr.speed_px_per_frame(), aspect))
        return results


def classify_person(speed, aspect, group_bump):
    if aspect < CROUCH_ASPECT_RATIO:
        return "threat"  # crouching / crawling posture
    if speed >= SPEED_THREAT_PX_PER_FRAME:
        return "threat"  # running / sudden fast motion
    if group_bump:
        return "threat"  # multiple people moving together
    if speed <= SPEED_NORMAL_PX_PER_FRAME:
        return "safe"    # stationary / loitering slowly
    return "normal"      # ordinary walking pace


# ---------------------------------------------------------------------------
# Model + tracker (loaded once, shared across frames)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


class YOLOProcessor:
    def __init__(self):
        self.model = load_model()
        self.tracker = CentroidTracker()
        self.frame_count = 0
        self.last_annotated = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % INFER_EVERY_N_FRAMES != 0 and self.last_annotated is not None:
            return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

        h0, w0 = img.shape[:2]
        scale = RESIZE_WIDTH / w0
        small = cv2.resize(img, (RESIZE_WIDTH, int(h0 * scale)))

        results = self.model.predict(small, conf=CONF_THRESHOLD, verbose=False)[0]
        t = time.time()

        person_dets = []
        other_boxes = []  # (bbox_in_orig_coords, label, conf)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # scale back to original frame size
            x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
            label = self.model.names[cls_id]

            if cls_id == COCO_PERSON_CLASS_ID:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                person_dets.append((cx, cy, (x1, y1, x2, y2)))
            else:
                other_boxes.append(((x1, y1, x2, y2), label, conf))

        tracked = self.tracker.update(person_dets, t)

        # group-movement bump: mark a person as "grouped" if another
        # tracked person is within GROUP_DISTANCE_PX
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b, _, _ in tracked]
        group_flags = []
        for i, (cx, cy) in enumerate(centers):
            grouped = False
            for j, (ox, oy) in enumerate(centers):
                if i == j:
                    continue
                if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < GROUP_DISTANCE_PX:
                    grouped = True
                    break
            group_flags.append(grouped)

        annotated = img.copy()

        for (bbox, label, conf) in other_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["object"], 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["object"], 2)

        for (tid, bbox, speed, aspect), grouped in zip(tracked, group_flags):
            status = classify_person(speed, aspect, grouped)
            x1, y1, x2, y2 = map(int, bbox)
            color = COLORS[status]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"person#{tid} [{status.upper()}]",
                        (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        self.last_annotated = annotated
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Live Camera Threat Detection", layout="wide")
st.title("Live Camera Tracking — Object Detection + Threat Status")
st.caption(
    "Open this page on your phone's browser and allow camera access. "
    "Each detected person is tagged SAFE / NORMAL / THREAT based on movement; "
    "other objects (chair, table, etc.) are boxed in cyan."
)

webrtc_streamer(
    key="live-camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(
    """
    ---
    **Legend:** 🟥 THREAT (fast motion, crouching, or group movement) ·
    🟨 NORMAL (regular walking) · 🟩 SAFE (stationary / very slow) ·
    🟦 other detected objects
    """
)

# ---------------------------------------------------------------------------
# SETUP NOTES (read before running)
# ---------------------------------------------------------------------------
#
# 1. Install extra deps (in your existing thermal_env):
#       pip install streamlit-webrtc av
#
# 2. IMPORTANT — camera permission requires a "secure context":
#    Phone browsers block camera access on plain http://<ip>:8501 URLs
#    (anything that isn't localhost or https). Easiest fix:
#
#       a) Install ngrok (https://ngrok.com/download), sign up free, run:
#            ngrok config add-authtoken <your-token>
#            ngrok http 8501
#          It prints an https://xxxx.ngrok-free.app URL — open THAT on
#          your phone instead of your laptop's local IP. This also works
#          even if your phone isn't on the same WiFi.
#
#       b) Alternative (same-WiFi only, no ngrok): generate a self-signed
#          cert with mkcert, then run:
#            streamlit run live_camera.py --server.sslCertfile cert.pem --server.sslKeyFile key.pem
#          and open https://<laptop-ip>:8501 on your phone, accepting the
#          browser's "unsafe certificate" warning.
#
# 3. Run:
#       streamlit run live_camera.py
#
# 4. If it feels laggy on your phone's data/wifi connection, try:
#       - increasing INFER_EVERY_N_FRAMES to 3 or 4
#       - lowering RESIZE_WIDTH to 480
#       - the RTX 4050 GPU will be used automatically by ultralytics if
#         torch+CUDA are installed in this env (same as your training setup)
