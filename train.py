from ultralytics import YOLO


def main():
    # Start from the small pretrained YOLOv8 model (downloads automatically the first time)
    model = YOLO("yolov8n.pt")

    # Fine-tune on your FLIR thermal person-detection dataset.
    # batch/imgsz lowered to reduce RAM usage on CPU-only training.
    model.train(
        data="data.yaml",   # must be in the same folder as this script
        epochs=25,
        imgsz=416,
        batch=4,
        workers=0,
        device="cpu",
        cache=False,
        name="flir_person_yolov8",
    )


if __name__ == "__main__":
    main()
