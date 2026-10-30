import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from backend.config import MODEL_PATH

class ThermalImageService:
    def __init__(self):
        self.model = None

    def get_model(self):
        if self.model is None:
            self.model = YOLO(MODEL_PATH)
        return self.model

    def enhance_thermal(self, frame_bgr):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process_image(self, image_bytes: bytes, conf: float = 0.25, apply_clahe: bool = False):
        model = self.get_model()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Invalid image file")

        if apply_clahe:
            img_bgr = self.enhance_thermal(img_bgr)

        # Convert to RGB for YOLO
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = model.predict(img_rgb, conf=conf, iou=0.4)
        result = results[0]

        # Generate plotted image
        plotted_bgr = result.plot()  # plotted is RGB in ultralytics plot
        # OpenCV encode to JPEG
        plotted_rgb = plotted_bgr[:, :, ::-1] if plotted_bgr.shape[2] == 3 else plotted_bgr
        _, buffer = cv2.imencode('.jpg', plotted_bgr)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        class_names = result.names
        person_detections = []
        other_detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                c = float(box.conf[0])
                label = class_names[cls_id]
                xyxy = [float(x) for x in box.xyxy[0].tolist()]

                item = {
                    "class": label,
                    "confidence": round(c, 4),
                    "confidence_pct": f"{round(c * 100, 1)}%",
                    "bbox": xyxy
                }
                if label == "person":
                    person_detections.append(item)
                else:
                    other_detections.append(item)

        return {
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "person_count": len(person_detections),
            "person_detections": person_detections,
            "other_detections": other_detections,
            "total_detections": len(person_detections) + len(other_detections)
        }

thermal_image_service = ThermalImageService()

