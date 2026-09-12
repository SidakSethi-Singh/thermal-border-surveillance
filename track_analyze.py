"""
Thermal person tracking + movement classification + false-alarm reduction.

Takes a video, runs YOLOv8 detection + ByteTrack tracking, and classifies
each tracked person's movement into one of 3 categories using simple,
explainable rules built on their motion history:

  - "crouching/crawling"  -> posture check (bounding box height/width ratio
                              drops well below that person's own baseline)
  - "group movement"      -> 2+ people close together, moving in the same
                              direction at similar speed
  - "normal walking"      -> default movement, or "stationary" if barely
                              moving at all

Thermal-specific handling:
  - CLAHE contrast enhancement is applied to each frame before detection,
    to counteract low contrast / thermal blooming common in IR footage.

False-alarm reduction (2 techniques):
  - Non-human classes (car, bicycle, etc.) are explicitly detected and
    labeled "non-threat (filtered)" in gray, instead of being ignored --
    showing the system actively distinguishes people from other heat
    sources rather than just not looking at them.
  - A person's suspicious behavior (crouching/group movement) must persist
    for several consecutive frames before it becomes a confirmed "ALERT" --
    a single noisy/glitched frame will not trigger a false alarm.

Multi-camera fusion (bonus, simulated):
  - True multi-physical-camera fusion (re-identifying a person across
    genuinely different camera angles) is a research-level problem, not
    buildable in a hackathon timeframe. This is a disclosed simulation:
    the single video frame is split into two overlapping "virtual camera"
    zones (left-coverage / right-coverage). When a tracked person is seen
    in the overlap zone, both virtual feeds are "reporting" the same
    target -- and that's flagged as cross-camera confirmed. This
    demonstrates the actual fusion decision logic the requirement is
    testing, on a single real camera feed.

Outputs an annotated video with bounding boxes, a movement label, and a
suspicion score per tracked person -- not just a terminal log.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "best.pt"
VIDEO_SOURCE = "input_video.mp4"       # <-- change this to your thermal video file
OUTPUT_PATH = "output_annotated.mp4"
CONF_THRESHOLD = 0.15

HISTORY_LEN = 15               # frames of motion history kept per tracked person
MIN_HISTORY_FOR_CLASSIFY = 5   # need at least this many frames before classifying
CROUCH_RATIO_DROP = 0.6        # flag "crouching" if h/w falls below 60% of baseline
GROUP_DIST_PX = 120            # max pixel distance to consider two people "together"
GROUP_DIR_COS = 0.6            # min direction-similarity (cosine) to count as "moving together"
SPEED_STATIONARY = 1.0         # px/frame below this counts as "stationary"

ALERT_CONFIRM_FRAMES = 8       # consecutive high-suspicion frames needed before a real "ALERT"
APPLY_CLAHE = True              # thermal contrast enhancement toggle

APPLY_MULTI_CAMERA_FUSION = True
CAM_A_COVERAGE = 0.65           # "camera A" covers left 0% - 65% of frame width
CAM_B_COVERAGE = 0.35           # "camera B" covers right 35% - 100% of frame width
                                 # -> overlap zone = 35%-65% of frame width, seen by both

SUSPICION = {
    "crouching/crawling": 0.85,
    "group movement": 0.55,
    "normal walking": 0.20,
    "stationary": 0.10,
    "observing": 0.10,  # not enough history yet to classify confidently
}
# -----------------------------------------

model = YOLO(MODEL_PATH)

track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
baseline_ratio = {}
alert_streak = defaultdict(int)  # consecutive high-suspicion frame count, per track
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def enhance_thermal(frame):
    """Apply CLAHE contrast enhancement to counteract thermal blooming / low contrast."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_camera_tag(cx, width):
    """Which virtual camera(s) currently see this x-position."""
    seen_by_a = cx <= width * CAM_A_COVERAGE
    seen_by_b = cx >= width * (1 - CAM_B_COVERAGE)
    if seen_by_a and seen_by_b:
        return "A+B"  # overlap zone -- fused, cross-camera confirmed
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


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Could not open video source: {VIDEO_SOURCE}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_idx = 0

    for result in model.track(
        source=VIDEO_SOURCE,
        conf=CONF_THRESHOLD,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
    ):
        frame = result.orig_img.copy()
        if APPLY_CLAHE:
            frame = enhance_thermal(frame)

        names = result.names
        current_boxes = {}
        non_human_boxes = []  # (x1, y1, x2, y2, class_name) -- filtered, not treated as threats

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

        # Draw non-human detections as explicitly filtered / non-threat (false-alarm reduction #1)
        for x1, y1, x2, y2, class_name in non_human_boxes:
            yellow = (0, 255, 255)  # bright, stands out against grayscale thermal footage
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
            # Fusion: a target confirmed by two overlapping virtual camera feeds
            # gets a small suspicion-confidence boost, reflecting higher certainty.
            if fused:
                suspicion = min(suspicion * 1.15, 1.0)

            is_high_suspicion = suspicion >= 0.5

            # Alert confirmation streak (false-alarm reduction #2): require several
            # consecutive high-suspicion frames before calling it a real ALERT
            if is_high_suspicion:
                alert_streak[track_id] += 1
            else:
                alert_streak[track_id] = 0

            confirmed_alert = alert_streak[track_id] >= ALERT_CONFIRM_FRAMES
            color = (0, 0, 255) if confirmed_alert else ((0, 165, 255) if is_high_suspicion else (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            short_label = {
                "crouching/crawling": "crouching",
                "group movement": "group",
                "normal walking": "walking",
                "stationary": "still",
                "observing": "...",
            }.get(label, label)
            prefix = "ALERT " if confirmed_alert else ""
            cam_suffix = f" [{cam_tag}]" if cam_tag else ""
            text = f"{prefix}#{track_id} {short_label} {suspicion:.2f}{cam_suffix}"
            cv2.putText(frame, text, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if APPLY_MULTI_CAMERA_FUSION:
            overlap_x1 = int(width * (1 - CAM_B_COVERAGE))
            overlap_x2 = int(width * CAM_A_COVERAGE)
            cv2.line(frame, (overlap_x1, 0), (overlap_x1, height), (255, 200, 0), 1)
            cv2.line(frame, (overlap_x2, 0), (overlap_x2, height), (255, 200, 0), 1)
            cv2.putText(frame, "Cam A", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            cv2.putText(frame, "Cam B", (width - 60, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            cv2.putText(frame, "overlap (fusion zone)", (overlap_x1 + 5, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

        out.write(frame)
        frame_idx += 1

    out.release()
    print(f"Done. Annotated video saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
