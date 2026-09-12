from ultralytics import YOLO


def main():
    # Load the last checkpoint saved before training was interrupted
    model = YOLO("runs/detect/flir_person_yolov8-3/weights/last.pt")

    # Resume picks up the exact same run: same epoch count, optimizer state, etc.
    model.train(resume=True)


if __name__ == "__main__":
    main()
