import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / "best.pt")
LIVE_MODEL_PATH = str(BASE_DIR / "yolov8s.pt")

MEDIA_DIR = BASE_DIR / "backend_media"
UPLOAD_DIR = MEDIA_DIR / "uploads"
OUTPUT_DIR = MEDIA_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Movement classification parameters
HISTORY_LEN = 15
MIN_HISTORY_FOR_CLASSIFY = 5
CROUCH_RATIO_DROP = 0.6
GROUP_DIST_PX = 120
GROUP_DIR_COS = 0.6
SPEED_STATIONARY = 1.0
ALERT_CONFIRM_FRAMES = 8

# Camera fusion parameters
APPLY_MULTI_CAMERA_FUSION = True
CAM_A_COVERAGE = 0.65
CAM_B_COVERAGE = 0.35

SUSPICION = {
    "crouching/crawling": 0.85,
    "group movement": 0.55,
    "normal walking": 0.20,
    "stationary": 0.10,
    "observing": 0.10,
}

# Live camera parameters
LIVE_CONF_THRESHOLD = 0.55
LIVE_INFER_EVERY_N_FRAMES = 2
LIVE_RESIZE_WIDTH = 640
LIVE_SPEED_THREAT = 28
LIVE_SPEED_NORMAL = 6
LIVE_CROUCH_ASPECT_RATIO = 0.9
LIVE_CROUCH_MIN_BOX_HEIGHT_FRAC = 0.55
LIVE_GROUP_DISTANCE_PX = 120
LIVE_TRACK_HISTORY_LEN = 8
LIVE_TRACK_MATCH_MAX_DIST = 60
LIVE_TRACK_STALE_SECONDS = 0.8
COCO_PERSON_CLASS_ID = 0

LIVE_COLORS_BGR = {
    "threat": (0, 0, 255),      # Red
    "normal": (0, 210, 255),    # Yellow/Orange
    "safe": (0, 200, 0),        # Green
    "object": (255, 200, 0),    # Blue/Cyan
}

