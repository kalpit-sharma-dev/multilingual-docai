import os
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess_image(img_path, out_dir):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Deskew
    img, angle = deskew(img)

    # Resize to max 2480x3508 while preserving aspect ratio
    max_w, max_h = 2480, 3508
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Normalize lighting (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Save processed image
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)

    # Return metadata
    return { "file": img_path, "output": out_path, "deskew_angle": angle }

def preprocess_dataset(raw_dir, out_dir, manifest_file="manifest.json"):
    os.makedirs(out_dir, exist_ok=True)
    all_imgs = glob(os.path.join(raw_dir, "*.png")) + glob(os.path.join(raw_dir, "*.jpg")) + glob(os.path.join(raw_dir, "*.jpeg"))

    manifest = []
    for img_path in tqdm(all_imgs, desc="Preprocessing images"):
        meta = preprocess_image(img_path, out_dir)
        if meta:
            manifest.append(meta)

    with open(os.path.join(out_dir, manifest_file), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Preprocessing complete. {len(manifest)} images processed. Manifest saved at {manifest_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess document images (deskew, resize, normalize)")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw image folder")
    parser.add_argument("--out", type=str, default="data/processed", help="Output folder for processed images")
    args = parser.parse_args()

    preprocess_dataset(args.raw, args.out)
