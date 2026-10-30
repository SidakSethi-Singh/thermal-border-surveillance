import os
import subprocess
import threading
import uuid
from collections import defaultdict, deque
import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

from backend.config import (
    MODEL_PATH,
    HISTORY_LEN,
    MIN_HISTORY_FOR_CLASSIFY,
    CROUCH_RATIO_DROP,
    GROUP_DIST_PX,
    GROUP_DIR_COS,
    SPEED_STATIONARY,
    ALERT_CONFIRM_FRAMES,
    APPLY_MULTI_CAMERA_FUSION,
    CAM_A_COVERAGE,
    CAM_B_COVERAGE,
    SUSPICION,
    UPLOAD_DIR,
    OUTPUT_DIR
)

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

class VideoTask:
    def __init__(self, task_id: str, input_path: str, output_path: str):
        self.task_id = task_id
        self.input_path = input_path
        self.output_path = output_path
        self.status = "queued"  # queued, processing, completed, error
        self.progress = 0.0
        self.alert_count = 0
        self.tracks_count = 0
        self.total_frames = 0
        self.processed_frames = 0
        self.duration_sec = 0.0
        self.fps = 20.0
        self.error_message = None
        self.result_url = None

        # Rich telemetry & classifications
        self.classification_breakdown = {
            "crouching": 0,
            "group": 0,
            "walking": 0,
            "stationary": 0,
            "non_threat_objects": 0,
            "fusion_overlap": 0
        }
        self.confirmed_alerts = []
        self.tracked_entities = []
        self.non_human_summary = {}

class ThermalVideoService:
    def __init__(self):
        self.model = None
        self.tasks: dict[str, VideoTask] = {}

    def get_model(self):
        if self.model is None:
            self.model = YOLO(MODEL_PATH)
        return self.model

    def enhance_thermal(self, frame_bgr, clahe):
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def get_camera_tag(self, cx, width):
        seen_by_a = cx <= width * CAM_A_COVERAGE
        seen_by_b = cx >= width * (1 - CAM_B_COVERAGE)
        if seen_by_a and seen_by_b:
            return "A+B"
        if seen_by_a:
            return "A"
        if seen_by_b:
            return "B"
        return "?"

    def velocity(self, history):
        if len(history) < 2:
            return 0.0, 0.0
        x0, y0, *_ = history[0]
        x1, y1, *_ = history[-1]
        n = len(history)
        return (x1 - x0) / n, (y1 - y0) / n

    def create_task(self, filename: str, content: bytes, conf_threshold: float = 0.15) -> VideoTask:
        task_id = str(uuid.uuid4())
        ext = os.path.splitext(filename)[1] or ".mp4"
        input_path = str(UPLOAD_DIR / f"{task_id}_input{ext}")
        output_path = str(OUTPUT_DIR / f"{task_id}_output.mp4")

        with open(input_path, "wb") as f:
            f.write(content)

        task = VideoTask(task_id, input_path, output_path)
        self.tasks[task_id] = task

        thread = threading.Thread(
            target=self._process_video_task,
            args=(task, conf_threshold),
            daemon=True
        )
        thread.start()
        return task

    def get_task(self, task_id: str) -> VideoTask | None:
        return self.tasks.get(task_id)

    def _process_video_task(self, task: VideoTask, conf_threshold: float):
        try:
            task.status = "processing"
            task.progress = 0.05

            model = self.get_model()
            track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
            baseline_ratio = {}
            alert_streak = defaultdict(int)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

            cap = cv2.VideoCapture(task.input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            if fps <= 0 or fps > 120:
                fps = 20.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            task.total_frames = total_frames
            task.fps = fps
            task.duration_sec = round(total_frames / fps, 2)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            raw_output_path = task.output_path.replace(".mp4", "_raw.mp4")
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
                            va = self.velocity(track_history[id_a])
                            vb = self.velocity(track_history[id_b])
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
                    vx, vy = self.velocity(history)
                    speed = (vx ** 2 + vy ** 2) ** 0.5
                    labels[track_id] = "stationary" if speed < SPEED_STATIONARY else "normal walking"
                return labels

            frame_idx = 0
            alert_count = 0
            all_tracked_ids = set()

            # Detailed telemetry accumulators
            track_first_seen = {}
            track_last_seen = {}
            track_behaviors = defaultdict(lambda: defaultdict(int))
            track_max_suspicion = defaultdict(float)
            track_alert_frames = defaultdict(int)
            track_cam_tags = defaultdict(set)
            non_human_counts = defaultdict(int)

            confirmed_alert_events = []
            active_alert_events = {}  # track_id -> current alert event

            for result in model.track(
                source=task.input_path, conf=conf_threshold, iou=0.4, persist=True,
                tracker="bytetrack.yaml", stream=True,
            ):
                frame = self.enhance_thermal(result.orig_img.copy(), clahe)
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
                            non_human_counts[class_name] += 1
                            continue

                        track_history[track_id].append((cx, cy, w, h, frame_idx))
                        current_boxes[int(track_id)] = (cx, cy, w, h)
                        all_tracked_ids.add(int(track_id))

                        if track_id not in track_first_seen:
                            track_first_seen[track_id] = frame_idx
                        track_last_seen[track_id] = frame_idx

                labels = classify_tracks(current_boxes)

                # Draw non-human boxes
                for x1, y1, x2, y2, class_name in non_human_boxes:
                    yellow = (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2)
                    cv2.putText(frame, f"{class_name} (non-threat)", (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, yellow, 1)

                # Draw and assess tracked persons
                for track_id, (cx, cy, w, h) in current_boxes.items():
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    label = labels.get(track_id, "observing")
                    suspicion = SUSPICION.get(label, 0.1)

                    cam_tag = self.get_camera_tag(cx, width) if APPLY_MULTI_CAMERA_FUSION else ""
                    fused = cam_tag == "A+B"
                    if fused:
                        suspicion = min(suspicion * 1.15, 1.0)
                        task.classification_breakdown["fusion_overlap"] += 1

                    track_cam_tags[track_id].add(cam_tag)
                    track_behaviors[track_id][label] += 1
                    track_max_suspicion[track_id] = max(track_max_suspicion[track_id], suspicion)

                    is_high = suspicion >= 0.5
                    alert_streak[track_id] = alert_streak[track_id] + 1 if is_high else 0
                    confirmed_alert = alert_streak[track_id] >= ALERT_CONFIRM_FRAMES

                    current_sec = round(frame_idx / fps, 2)

                    if confirmed_alert:
                        alert_count += 1
                        track_alert_frames[track_id] += 1

                        if track_id not in active_alert_events:
                            alert_event = {
                                "alert_id": len(confirmed_alert_events) + 1,
                                "track_id": int(track_id),
                                "behavior": label,
                                "suspicion": round(suspicion, 2),
                                "cam_tag": cam_tag,
                                "start_frame": frame_idx,
                                "start_time_sec": current_sec,
                                "start_time_formatted": format_timestamp(current_sec),
                                "duration_frames": 1,
                                "severity": "CRITICAL" if suspicion >= 0.8 else "ELEVATED"
                            }
                            active_alert_events[track_id] = alert_event
                            confirmed_alert_events.append(alert_event)
                        else:
                            active_alert_events[track_id]["duration_frames"] += 1
                            active_alert_events[track_id]["suspicion"] = max(
                                active_alert_events[track_id]["suspicion"], round(suspicion, 2)
                            )
                    else:
                        if track_id in active_alert_events:
                            del active_alert_events[track_id]

                    color = (0, 0, 255) if confirmed_alert else ((0, 165, 255) if is_high else (0, 255, 0))

                    # Thick high-contrast tactical bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Bounding box corner brackets for high-tech HUD look
                    bracket_len = int(min(15, float(w) / 4.0, float(h) / 4.0))
                    if bracket_len > 3:
                        cv2.line(frame, (int(x1), int(y1)), (int(x1 + bracket_len), int(y1)), (255, 255, 255), 2)
                        cv2.line(frame, (int(x1), int(y1)), (int(x1), int(y1 + bracket_len)), (255, 255, 255), 2)
                        cv2.line(frame, (int(x2), int(y1)), (int(x2 - bracket_len), int(y1)), (255, 255, 255), 2)
                        cv2.line(frame, (int(x2), int(y1)), (int(x2), int(y1 + bracket_len)), (255, 255, 255), 2)

                    short_label = {
                        "crouching/crawling": "CROUCH", "group movement": "GROUP",
                        "normal walking": "WALK", "stationary": "STILL", "observing": "OBS",
                    }.get(label, label.upper())
                    prefix = "ALERT " if confirmed_alert else ""
                    cam_suffix = f" [{cam_tag}]" if cam_tag else ""

                    tag_text = f"{prefix}#{track_id} {short_label} {int(suspicion * 100)}%{cam_suffix}"
                    # Label background pill
                    box_w = int(len(tag_text) * 8.5)
                    cv2.rectangle(frame, (int(x1), int(max(y1 - 18, 0))), (int(x1 + box_w), int(max(y1, 18))), (15, 15, 15), -1)
                    cv2.putText(frame, tag_text, (int(x1 + 2), int(max(y1 - 4, 14))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

                if APPLY_MULTI_CAMERA_FUSION:
                    overlap_x1 = int(width * (1 - CAM_B_COVERAGE))
                    overlap_x2 = int(width * CAM_A_COVERAGE)
                    # Boundary lines
                    cv2.line(frame, (int(overlap_x1), 0), (int(overlap_x1), int(height)), (0, 229, 255), 1)
                    cv2.line(frame, (int(overlap_x2), 0), (int(overlap_x2), int(height)), (0, 229, 255), 1)
                    cv2.putText(frame, "CAM A COVERAGE", (12, int(height) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1)
                    cv2.putText(frame, "FUSION OVERLAP [A+B]", (int(overlap_x1) + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 255), 1)
                    cv2.putText(frame, "CAM B COVERAGE", (int(width) - 140, int(height) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1)

                out.write(frame)
                frame_idx += 1
                task.processed_frames = frame_idx
                task.progress = min(0.1 + (frame_idx / total_frames) * 0.8, 0.9)

            out.release()

            # Compile classified tracked entities list
            entities_list = []
            crouch_total = 0
            group_total = 0
            walk_total = 0
            still_total = 0

            for tid in sorted(all_tracked_ids):
                behaviors = track_behaviors[tid]
                dominant_b = max(behaviors.keys(), key=lambda k: behaviors[k]) if behaviors else "normal walking"
                if dominant_b == "crouching/crawling":
                    crouch_total += 1
                elif dominant_b == "group movement":
                    group_total += 1
                elif dominant_b == "stationary":
                    still_total += 1
                else:
                    walk_total += 1

                first_sec = round(track_first_seen.get(tid, 0) / fps, 1)
                last_sec = round(track_last_seen.get(tid, 0) / fps, 1)
                duration = round(max(0.1, last_sec - first_sec), 1)

                is_alert = track_alert_frames[tid] > 0
                entities_list.append({
                    "track_id": int(tid),
                    "dominant_behavior": dominant_b,
                    "max_suspicion": round(track_max_suspicion[tid], 2),
                    "max_suspicion_pct": f"{int(track_max_suspicion[tid] * 100)}%",
                    "confirmed_alert": is_alert,
                    "alert_frames": track_alert_frames[tid],
                    "camera_tags": sorted(list(track_cam_tags[tid])),
                    "first_seen_sec": first_sec,
                    "first_seen_formatted": format_timestamp(first_sec),
                    "last_seen_sec": last_sec,
                    "last_seen_formatted": format_timestamp(last_sec),
                    "duration_sec": duration
                })

            task.classification_breakdown = {
                "crouching": crouch_total,
                "group": group_total,
                "walking": walk_total,
                "stationary": still_total,
                "non_threat_objects": sum(non_human_counts.values()),
                "fusion_overlap": task.classification_breakdown["fusion_overlap"]
            }
            task.confirmed_alerts = confirmed_alert_events
            task.tracked_entities = sorted(entities_list, key=lambda x: (not x["confirmed_alert"], -x["max_suspicion"]))
            task.non_human_summary = dict(non_human_counts)

            # Transcode with ffmpeg
            task.progress = 0.92
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg_exe, "-y", "-i", raw_output_path,
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    task.output_path
                ],
                check=True, capture_output=True,
            )
            if os.path.exists(raw_output_path):
                os.remove(raw_output_path)
            if os.path.exists(task.input_path):
                os.remove(task.input_path)

            task.alert_count = alert_count
            task.tracks_count = len(all_tracked_ids)
            task.progress = 1.0
            task.status = "completed"
            task.result_url = f"/api/detect/video/result/{task.task_id}"

        except Exception as e:
            task.status = "error"
            task.error_message = str(e)

thermal_video_service = ThermalVideoService()
