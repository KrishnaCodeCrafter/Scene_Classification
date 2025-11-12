import os, cv2, shutil, random, warnings
from PIL import Image
import imagehash
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

# =======================================================
# PIL safety setup
# =======================================================
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None
MAX_DIM = 8000  # resize very large images

# =======================================================
# CONFIG
# =======================================================
RAW_DIR = "./scene_dataset"
OUTPUT_CLEAN_DIR = "./scene_dataset_clean"
FINAL_SPLIT_DIR = "./scene_dataset_final"

MIN_BLUR = 120.0
MIN_BRIGHT = 30
MAX_BRIGHT = 230
MIN_ASPECT = 0.6
MAX_ASPECT = 2.0
USE_CLIP = True
CLIP_THRESHOLD = 0.25
SPLIT_RATIOS = (0.7, 0.15, 0.15)

# =======================================================
# CLIP setup
# =======================================================
if USE_CLIP:
    print("🔁 Loading CLIP model for semantic relevance filtering...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", use_fast=True
    )

# =======================================================
# Helpers
# =======================================================
def resize_if_huge(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w > MAX_DIM or h > MAX_DIM:
                img.thumbnail((MAX_DIM, MAX_DIM))
                img.save(path)
    except Exception:
        pass

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()

def aspect_ratio(img):
    h, w = img.shape[:2]
    return w / h

def clip_relevance(img_path, label_text):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=[label_text], images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        score = outputs.logits_per_image.softmax(dim=1)[0, 0].item()
        return score
    except Exception:
        return 0.0

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# =======================================================
# Cleaning
# =======================================================
def clean_all_splits():
    seen_hashes = set()
    ensure_dir(OUTPUT_CLEAN_DIR)

    for split in ["train", "val", "test"]:
        split_path = os.path.join(RAW_DIR, split)
        if not os.path.exists(split_path):
            continue
        print(f"\n🧹 Cleaning {split} split...")

        for label in os.listdir(split_path):
            label_in = os.path.join(split_path, label)
            if not os.path.isdir(label_in):
                continue
            label_out = os.path.join(OUTPUT_CLEAN_DIR, label)
            os.makedirs(label_out, exist_ok=True)

            kept, removed = 0, 0
            for fname in tqdm(os.listdir(label_in), desc=f"{label}"):
                if not fname.lower().endswith(".jpg"):
                    continue
                src = os.path.join(label_in, fname)

                try:
                    resize_if_huge(src)
                    img = cv2.imread(src)
                    if img is None:
                        removed += 1; continue

                    if blur_score(img) < MIN_BLUR:
                        removed += 1; continue
                    bright = brightness_score(img)
                    if bright < MIN_BRIGHT or bright > MAX_BRIGHT:
                        removed += 1; continue
                    ar = aspect_ratio(img)
                    if ar < MIN_ASPECT or ar > MAX_ASPECT:
                        removed += 1; continue

                    ph = imagehash.average_hash(Image.open(src))
                    if ph in seen_hashes:
                        removed += 1; continue
                    seen_hashes.add(ph)

                    if USE_CLIP:
                        score = clip_relevance(src, label.replace("_", " "))
                        if score < CLIP_THRESHOLD:
                            removed += 1; continue

                    shutil.copy(src, os.path.join(label_out, fname))
                    kept += 1
                except Exception:
                    removed += 1
                    continue

            print(f"✅ {label}: kept {kept}, removed {removed}")

# =======================================================
# Re-split
# =======================================================
def rebuild_splits():
    print("\n⚖️ Re-splitting cleaned dataset evenly...")
    ensure_dir(FINAL_SPLIT_DIR)
    random.seed(42)

    for label in os.listdir(OUTPUT_CLEAN_DIR):
        label_path = os.path.join(OUTPUT_CLEAN_DIR, label)
        if not os.path.isdir(label_path):
            continue

        files = [f for f in os.listdir(label_path) if f.endswith(".jpg")]
        random.shuffle(files)
        n = len(files)
        n_train = int(SPLIT_RATIOS[0] * n)
        n_val = int(SPLIT_RATIOS[1] * n)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for split, file_list in splits.items():
            out_dir = os.path.join(FINAL_SPLIT_DIR, split, label)
            ensure_dir(out_dir)
            for f in file_list:
                shutil.copy(os.path.join(label_path, f), os.path.join(out_dir, f))

        print(f"📂 {label}: {n_train} train, {n_val} val, {n - n_train - n_val} test")

    print("\n🎯 Final balanced dataset created in:", FINAL_SPLIT_DIR)

# =======================================================
# MAIN
# =======================================================
if __name__ == "__main__":
    clean_all_splits()
    rebuild_splits()
    print("\n✅ Dataset cleaned, resized, deduplicated, and re-split successfully!")
