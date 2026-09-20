import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time
import subprocess
import imageio_ffmpeg
from collections import defaultdict, deque

import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Thermal Person Detector", layout="centered")

MODEL_PATH = "best.pt"
LIVE_MODEL_PATH = "yolov8s.pt"  # general COCO model for the Live Camera tab (auto-downloads)

# ---- movement classification config (same logic as track_analyze.py) ----
HISTORY_LEN = 15
MIN_HISTORY_FOR_CLASSIFY = 5
CROUCH_RATIO_DROP = 0.6
GROUP_DIST_PX = 120
GROUP_DIR_COS = 0.6
SPEED_STATIONARY = 1.0
ALERT_CONFIRM_FRAMES = 8

APPLY_MULTI_CAMERA_FUSION = True
CAM_A_COVERAGE = 0.65
CAM_B_COVERAGE = 0.35  # overlap zone = 35%-65% of frame width, seen by both virtual cameras

SUSPICION = {
    "crouching/crawling": 0.85,
    "group movement": 0.55,
    "normal walking": 0.20,
    "stationary": 0.10,
    "observing": 0.10,
}

# ---- Live Camera tab config ----
LIVE_CONF_THRESHOLD = 0.55         # raised from 0.4 -> fewer false positives (e.g. bag -> person)
LIVE_INFER_EVERY_N_FRAMES = 2
LIVE_RESIZE_WIDTH = 640
LIVE_SPEED_THREAT = 28
LIVE_SPEED_NORMAL = 6
LIVE_CROUCH_ASPECT_RATIO = 0.9
LIVE_CROUCH_MIN_BOX_HEIGHT_FRAC = 0.55  # only trust the crouch/aspect heuristic when most of the body is in frame
LIVE_GROUP_DISTANCE_PX = 120
LIVE_TRACK_HISTORY_LEN = 8
LIVE_TRACK_MATCH_MAX_DIST = 60      # tightened from 80 -> less chance of a duplicate ID
LIVE_TRACK_STALE_SECONDS = 0.8      # shortened from 1.5 -> stale ghost track drops faster
COCO_PERSON_CLASS_ID = 0
LIVE_COLORS_BGR = {
    "threat": (0, 0, 255),
    "normal": (0, 210, 255),
    "safe": (0, 200, 0),
    "object": (255, 200, 0),
}
LIVE_RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_live_model():
    return YOLO(LIVE_MODEL_PATH)


def enhance_thermal(frame, clahe):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_camera_tag(cx, width):
    seen_by_a = cx <= width * CAM_A_COVERAGE
    seen_by_b = cx >= width * (1 - CAM_B_COVERAGE)
    if seen_by_a and seen_by_b:
        return "A+B"
    if seen_by_a:
        return "A"
    if seen_by_b:
        return "B"
    return "?"


def velocity(history):
    if len(history) < 2:
        return 0.0, 0.0
    x0, y0, *_ = history[0]
    x1, y1, *_ = history[-1]
    n = len(history)
    return (x1 - x0) / n, (y1 - y0) / n


def process_video(model, input_path, output_path, conf_threshold=0.15, progress_callback=None):
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    baseline_ratio = {}
    alert_streak = defaultdict(int)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_output_path = output_path.replace(".mp4", "_raw.mp4")
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    def is_crouching(track_id, history):
        cx, cy, w, h, _ = history[-1]
        ratio = h / max(w, 1)
        if track_id not in baseline_ratio and len(history) >= MIN_HISTORY_FOR_CLASSIFY:
            early = list(history)[:3]
            baseline_ratio[track_id] = float(np.mean([hh / max(ww, 1) for (_, _, ww, hh, _) in early]))
        base = baseline_ratio.get(track_id)
        return bool(base and ratio < base * CROUCH_RATIO_DROP)

    def find_grouped_ids(current_boxes):
        ids = list(current_boxes.keys())
        grouped = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                cx_a, cy_a, *_ = current_boxes[id_a]
                cx_b, cy_b, *_ = current_boxes[id_b]
                dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
                if dist < GROUP_DIST_PX:
                    va = velocity(track_history[id_a])
                    vb = velocity(track_history[id_b])
                    mag_a = (va[0] ** 2 + va[1] ** 2) ** 0.5
                    mag_b = (vb[0] ** 2 + vb[1] ** 2) ** 0.5
                    if mag_a > 0.3 and mag_b > 0.3:
                        cos_sim = (va[0] * vb[0] + va[1] * vb[1]) / (mag_a * mag_b + 1e-6)
                        if cos_sim > GROUP_DIR_COS:
                            grouped.add(id_a)
                            grouped.add(id_b)
        return grouped

    def classify_tracks(current_boxes):
        labels = {}
        grouped = find_grouped_ids(current_boxes)
        for track_id in current_boxes:
            history = track_history[track_id]
            if len(history) < MIN_HISTORY_FOR_CLASSIFY:
                labels[track_id] = "observing"
                continue
            if is_crouching(track_id, history):
                labels[track_id] = "crouching/crawling"
                continue
            if track_id in grouped:
                labels[track_id] = "group movement"
                continue
            vx, vy = velocity(history)
            speed = (vx ** 2 + vy ** 2) ** 0.5
            labels[track_id] = "stationary" if speed < SPEED_STATIONARY else "normal walking"
        return labels

    frame_idx = 0
    alert_count = 0

    for result in model.track(
        source=input_path, conf=conf_threshold, iou=0.4, persist=True,
        tracker="bytetrack.yaml", stream=True,
    ):
        frame = enhance_thermal(result.orig_img.copy(), clahe)
        names = result.names
        current_boxes = {}
        non_human_boxes = []

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, cls_id in zip(
                result.boxes.xywh.cpu().numpy(),
                result.boxes.id.cpu().numpy().astype(int),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                cx, cy, w, h = box
                class_name = names[cls_id]
                if class_name != "person":
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    non_human_boxes.append((x1, y1, x2, y2, class_name))
                    continue
                track_history[track_id].append((cx, cy, w, h, frame_idx))
                current_boxes[int(track_id)] = (cx, cy, w, h)

        labels = classify_tracks(current_boxes)

        for x1, y1, x2, y2, class_name in non_human_boxes:
            yellow = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2)
            cv2.putText(frame, f"{class_name} (non-threat)", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1)

        for track_id, (cx, cy, w, h) in current_boxes.items():
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            label = labels.get(track_id, "observing")
            suspicion = SUSPICION.get(label, 0.1)

            cam_tag = get_camera_tag(cx, width) if APPLY_MULTI_CAMERA_FUSION else ""
            fused = cam_tag == "A+B"
            if fused:
                suspicion = min(suspicion * 1.15, 1.0)

            is_high = suspicion >= 0.5

            alert_streak[track_id] = alert_streak[track_id] + 1 if is_high else 0
            confirmed_alert = alert_streak[track_id] >= ALERT_CONFIRM_FRAMES
            if confirmed_alert:
                alert_count += 1
            color = (0, 0, 255) if confirmed_alert else ((0, 165, 255) if is_high else (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            short_label = {
                "crouching/crawling": "crouching", "group movement": "group",
                "normal walking": "walking", "stationary": "still", "observing": "...",
            }.get(label, label)
            prefix = "ALERT " if confirmed_alert else ""
            cam_suffix = f" [{cam_tag}]" if cam_tag else ""
            cv2.putText(frame, f"{prefix}#{track_id} {short_label} {suspicion:.2f}{cam_suffix}",
                        (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if APPLY_MULTI_CAMERA_FUSION:
            overlap_x1 = int(width * (1 - CAM_B_COVERAGE))
            overlap_x2 = int(width * CAM_A_COVERAGE)
            cv2.line(frame, (overlap_x1, 0), (overlap_x1, height), (255, 200, 0), 1)
            cv2.line(frame, (overlap_x2, 0), (overlap_x2, height), (255, 200, 0), 1)
            cv2.putText(frame, "Cam A", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            cv2.putText(frame, "Cam B", (width - 60, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        out.write(frame)
        frame_idx += 1
        if progress_callback and total_frames:
            progress_callback(min(frame_idx / total_frames, 1.0))

    out.release()

    # Re-encode with ffmpeg into a browser-compatible H.264 mp4 (OpenCV's own
    # output often isn't playable in a web <video> element, even though it
    # opens fine in a desktop video player).
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", raw_output_path,
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         output_path],
        check=True, capture_output=True,
    )
    os.remove(raw_output_path)

    return alert_count


# =============================================================================
# Live Camera tab: centroid tracker + movement -> threat/normal/safe
# =============================================================================

class LivePersonTrack:
    __slots__ = ("id", "positions", "last_seen", "last_bbox")

    def __init__(self, track_id, cx, cy, bbox, t):
        self.id = track_id
        self.positions = deque(maxlen=LIVE_TRACK_HISTORY_LEN)
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
        (x0, y0, _), (x1, y1, _) = self.positions[-2], self.positions[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


class LiveCentroidTracker:
    def __init__(self):
        self.tracks = {}
        self._next_id = 1

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections, t):
        stale = [tid for tid, tr in self.tracks.items() if t - tr.last_seen > LIVE_TRACK_STALE_SECONDS]
        for tid in stale:
            del self.tracks[tid]

        unmatched = list(range(len(detections)))
        for tid, tr in list(self.tracks.items()):
            if not unmatched:
                break
            last_cx, last_cy, _ = tr.positions[-1]
            best_j, best_dist = None, LIVE_TRACK_MATCH_MAX_DIST
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
            self.tracks[tid] = LivePersonTrack(tid, cx, cy, bbox, t)

        results = []
        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = tr.last_bbox
            w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            results.append((tid, tr.last_bbox, tr.speed_px_per_frame(), h / w))
        return results


def classify_live_person(speed, aspect, group_bump, box_height_frac):
    # Only trust the crouch/aspect heuristic when most of the body is
    # actually in frame (full-body CCTV-style shot). A close-up webcam
    # framing naturally produces a wide/square box for a standing person,
    # which used to get misread as "crouching".
    if box_height_frac > LIVE_CROUCH_MIN_BOX_HEIGHT_FRAC and aspect < LIVE_CROUCH_ASPECT_RATIO:
        return "threat"
    if speed >= LIVE_SPEED_THREAT:
        return "threat"
    if group_bump:
        return "threat"
    if speed <= LIVE_SPEED_NORMAL:
        return "safe"
    return "normal"


class LiveProcessor:
    def __init__(self):
        self.model = load_live_model()
        self.tracker = LiveCentroidTracker()
        self.frame_count = 0
        self.last_annotated = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % LIVE_INFER_EVERY_N_FRAMES != 0 and self.last_annotated is not None:
            return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

        h0, w0 = img.shape[:2]
        scale = LIVE_RESIZE_WIDTH / w0
        small = cv2.resize(img, (LIVE_RESIZE_WIDTH, int(h0 * scale)))

        results = self.model.predict(small, conf=LIVE_CONF_THRESHOLD, verbose=False)[0]
        t = time.time()

        person_dets, other_boxes = [], []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
            label = self.model.names[cls_id]
            if cls_id == COCO_PERSON_CLASS_ID:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                person_dets.append((cx, cy, (x1, y1, x2, y2)))
            else:
                other_boxes.append(((x1, y1, x2, y2), label, conf))

        tracked = self.tracker.update(person_dets, t)
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b, _, _ in tracked]
        group_flags = []
        for i, (cx, cy) in enumerate(centers):
            grouped = any(
                i != j and ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < LIVE_GROUP_DISTANCE_PX
                for j, (ox, oy) in enumerate(centers)
            )
            group_flags.append(grouped)

        annotated = img.copy()
        for (bbox, label, conf) in other_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), LIVE_COLORS_BGR["object"], 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIVE_COLORS_BGR["object"], 2)

        for (tid, bbox, speed, aspect), grouped in zip(tracked, group_flags):
            x1, y1, x2, y2 = bbox
            box_height_frac = (y2 - y1) / max(h0, 1)
            status = classify_live_person(speed, aspect, grouped, box_height_frac)
            x1, y1, x2, y2 = map(int, bbox)
            color = LIVE_COLORS_BGR[status]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"person#{tid} [{status.upper()}]",
                        (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        self.last_annotated = annotated
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------- UI ----------------
st.title("Thermal Person Detection & Border Surveillance")

mode = st.radio("Choose input type", ["Image", "Video", "Live Camera"], horizontal=True)

if mode == "Image":
    model = load_model()
    st.write("Upload a thermal image to detect people in it.")
    uploaded_file = st.file_uploader("Choose a thermal image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Running detection..."):
            results = model.predict(np.array(image), conf=0.25, iou=0.4)

        annotated = results[0].plot()[:, :, ::-1]
        st.image(annotated, caption="Detections (all classes)", width="stretch")

        class_names = results[0].names
        person_confidences = [
            float(box.conf[0]) for box in results[0].boxes
            if class_names[int(box.cls[0])] == "person"
        ]

        st.subheader("Detections")
        st.success(f"Detected {len(person_confidences)} person(s)")
        if person_confidences:
            st.subheader("Person confidence scores")
            for i, conf in enumerate(person_confidences):
                st.write(f"Person {i + 1}: {conf:.2%} confidence")

elif mode == "Video":
    model = load_model()
    st.write("Upload a short video to run tracking, movement classification, and alert detection.")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            input_path = tmp_in.name

        output_path = input_path.replace(".mp4", "_out.mp4")

        progress_bar = st.progress(0.0, text="Processing video...")

        def update_progress(pct):
            progress_bar.progress(pct, text=f"Processing video... {int(pct * 100)}%")

        with st.spinner("Running detection, tracking, and classification..."):
            alert_count = process_video(model, input_path, output_path, progress_callback=update_progress)

        progress_bar.empty()

        st.subheader("Result")
        if alert_count > 0:
            st.error(f"{alert_count} confirmed ALERT(s) detected during this video")
        else:
            st.success("No confirmed alerts — only normal activity detected")

        st.video(output_path)
        st.caption("Yellow = non-human (filtered) | Green = normal | Orange = elevated | Red = confirmed ALERT")
        st.caption("[A]/[B]/[A+B] = simulated virtual camera coverage; [A+B] means confirmed by both overlapping feeds")

        os.unlink(input_path)

else:  # Live Camera
    st.write("Open this page on your phone's browser (via the ngrok HTTPS link) and allow camera access.")
    st.caption(
        "Uses a general object model (not the thermal one) so it can detect everyday objects "
        "like chairs and tables too. Detected people are tagged SAFE / NORMAL / THREAT based on movement."
    )
    webrtc_streamer(
        key="live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=LIVE_RTC_CONFIGURATION,
        video_processor_factory=LiveProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("🟥 THREAT (fast motion / crouching / group) · 🟨 NORMAL (walking) · 🟩 SAFE (stationary) · 🟦 other objects")
