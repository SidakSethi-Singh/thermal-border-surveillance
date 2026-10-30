import base64
import time
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import (
    LIVE_MODEL_PATH,
    LIVE_CONF_THRESHOLD,
    LIVE_INFER_EVERY_N_FRAMES,
    LIVE_RESIZE_WIDTH,
    LIVE_SPEED_THREAT,
    LIVE_SPEED_NORMAL,
    LIVE_CROUCH_ASPECT_RATIO,
    LIVE_CROUCH_MIN_BOX_HEIGHT_FRAC,
    LIVE_GROUP_DISTANCE_PX,
    LIVE_TRACK_HISTORY_LEN,
    LIVE_TRACK_MATCH_MAX_DIST,
    LIVE_TRACK_STALE_SECONDS,
    COCO_PERSON_CLASS_ID,
    LIVE_COLORS_BGR
)

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
    if box_height_frac > LIVE_CROUCH_MIN_BOX_HEIGHT_FRAC and aspect < LIVE_CROUCH_ASPECT_RATIO:
        return "threat"
    if speed >= LIVE_SPEED_THREAT:
        return "threat"
    if group_bump:
        return "threat"
    if speed <= LIVE_SPEED_NORMAL:
        return "safe"
    return "normal"


class LiveStreamProcessor:
    def __init__(self):
        self.model = None
        self.tracker = LiveCentroidTracker()
        self.frame_count = 0
        self.last_annotated = None

    def get_model(self):
        if self.model is None:
            self.model = YOLO(LIVE_MODEL_PATH)
        return self.model

    def process_frame(self, frame_bgr, conf_threshold: float = LIVE_CONF_THRESHOLD):
        model = self.get_model()
        self.frame_count += 1
        h0, w0 = frame_bgr.shape[:2]
        scale = LIVE_RESIZE_WIDTH / max(w0, 1)
        small = cv2.resize(frame_bgr, (LIVE_RESIZE_WIDTH, int(h0 * scale)))

        results = model.predict(small, conf=conf_threshold, verbose=False)[0]
        t = time.time()

        person_dets = []
        other_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
            label = model.names[cls_id]
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

        annotated = frame_bgr.copy()
        for (bbox, label, conf) in other_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), LIVE_COLORS_BGR["object"], 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIVE_COLORS_BGR["object"], 2)

        telemetry_tracks = []
        threat_count = 0

        for (tid, bbox, speed, aspect), grouped in zip(tracked, group_flags):
            x1, y1, x2, y2 = bbox
            box_height_frac = (y2 - y1) / max(h0, 1)
            status = classify_live_person(speed, aspect, grouped, box_height_frac)
            if status == "threat":
                threat_count += 1

            x1_i, y1_i, x2_i, y2_i = map(int, bbox)
            color = LIVE_COLORS_BGR[status]
            cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), color, 2)
            cv2.putText(annotated, f"person#{tid} [{status.upper()}]",
                        (x1_i, max(y1_i - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            telemetry_tracks.append({
                "id": tid,
                "status": status,
                "speed": round(speed, 1),
                "aspect": round(aspect, 2),
                "grouped": grouped,
                "bbox": [x1_i, y1_i, x2_i, y2_i]
            })

        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "image": f"data:image/jpeg;base64,{annotated_b64}",
            "telemetry": {
                "active_persons": len(telemetry_tracks),
                "threat_count": threat_count,
                "other_objects": len(other_boxes),
                "tracks": telemetry_tracks
            }
        }

live_processor = LiveStreamProcessor()

