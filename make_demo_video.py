"""
Stitches a folder of images into a short synthetic video, so the
tracking/classification pipeline has something to run on.

Note: since these are separate photos (not real consecutive video frames),
people will "jump" between positions rather than move smoothly. This is
fine for demonstrating that the pipeline works -- just be upfront about it
being a synthesized demo video, not real footage.
"""

import cv2
import glob
import os

# ---------------- CONFIG ----------------
IMAGE_FOLDER = "valid/images"      # folder of thermal images to use
OUTPUT_PATH = "input_video.mp4"    # matches VIDEO_SOURCE in track_analyze.py
NUM_IMAGES = 40                    # how many images to include
FPS = 5                            # low fps since images aren't truly sequential
# -----------------------------------------


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))[:NUM_IMAGES]

    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}")
        return

    first = cv2.imread(image_paths[0])
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (width, height))

    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (width, height))
        # repeat each frame a few times so tracking has enough frames to work with
        for _ in range(6):
            out.write(img)

    out.release()
    print(f"Done. Synthetic video saved to {OUTPUT_PATH} ({len(image_paths)} source images)")


if __name__ == "__main__":
    main()
