from ultralytics import YOLO


def main():
    # Start from the small pretrained YOLOv8 model (downloads automatically the first time)
    model = YOLO("yolov8n.pt")

    # Fine-tune on your FLIR thermal person-detection dataset.
    # GPU training (RTX 4050) — higher batch, higher resolution, more epochs than the CPU run.
    model.train(
        data="data.yaml",       # must be in the same folder as this script
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,
        device=0,               # explicitly use GPU 0 (your RTX 4050)
        cache=False,              # cache images in RAM for faster epochs (you have 24GB RAM)
        name="flir_person_yolov8_gpu",   # new run name — won't overwrite your original results
    )


if __name__ == "__main__":
    main()