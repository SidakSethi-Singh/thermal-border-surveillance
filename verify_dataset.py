"""
Quick sanity-check script for a YOLOv8-format dataset (Roboflow export).

Usage:
    python verify_dataset.py /path/to/dataset/train --n 8

It expects the standard YOLOv8 layout:
    dataset/
      train/
        images/
        labels/
      data.yaml   (optional, used to get class names)

For each sampled image it draws the bounding boxes from the matching
.txt label file and saves the result into an "verify_out/" folder next
to the split you pointed at, so you can quickly flip through them.
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import yaml


def load_class_names(dataset_root: Path):
    """Look for data.yaml in the dataset root (one level above the split) and
    return its class name list, or None if not found."""
    for candidate in [dataset_root / "data.yaml", dataset_root.parent / "data.yaml"]:
        if candidate.exists():
            with open(candidate, "r") as f:
                data = yaml.safe_load(f)
            names = data.get("names")
            if isinstance(names, dict):
                # some exports use {0: 'person', 1: 'car', ...}
                return [names[i] for i in sorted(names)]
            return names
    return None


def draw_boxes(image_path: Path, label_path: Path, class_names, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [skip] could not read image: {image_path}")
        return
    h, w = img.shape[:2]

    if not label_path.exists():
        print(f"  [warn] no label file for {image_path.name}")
        cv2.imwrite(str(out_path), img)
        return

    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])

        # YOLO format is normalized center-x, center-y, width, height
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        color = (0, 255, 0) if label.lower().startswith("person") else (255, 0, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check a YOLOv8-format dataset split.")
    parser.add_argument("split_dir", type=str,
                         help="Path to the split folder, e.g. dataset/train (must contain images/ and labels/)")
    parser.add_argument("--n", type=int, default=8, help="Number of random samples to check")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to both exist. "
                          f"Point this at a split folder like dataset/train.")

    class_names = load_class_names(split_dir.parent)
    if class_names:
        print(f"Loaded class names: {class_names}")
    else:
        print("No data.yaml found — boxes will be labeled with raw class IDs instead of names.")

    all_images = sorted([p for p in images_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not all_images:
        raise SystemExit(f"No images found in {images_dir}")

    n = min(args.n, len(all_images))
    sample = random.sample(all_images, n)

    out_dir = split_dir.parent / "verify_out"
    out_dir.mkdir(exist_ok=True)

    print(f"\nChecking {n} random images out of {len(all_images)} total...\n")

    person_count = 0
    total_boxes = 0
    for img_path in sample:
        label_path = labels_dir / (img_path.stem + ".txt")
        out_path = out_dir / img_path.name
        draw_boxes(img_path, label_path, class_names, out_path)
        print(f"  ok -> {out_path}")

        if label_path.exists():
            with open(label_path) as f:
                for ln in f:
                    parts = ln.split()
                    if not parts:
                        continue
                    total_boxes += 1
                    cls_id = int(float(parts[0]))
                    if class_names and cls_id < len(class_names) and "person" in class_names[cls_id].lower():
                        person_count += 1

    print(f"\nDone. Saved annotated samples to: {out_dir}")
    print(f"Boxes seen in sample: {total_boxes} total, {person_count} labeled 'person'.")
    print("Open the images in verify_out/ and confirm the green boxes actually sit on people.")


if __name__ == "__main__":
    main()
