import os
import time
import random
import shutil
import requests
from PIL import Image
import imagehash
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
load_dotenv()
UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")

SCENE_LABELS = {
    "Savanna_Grassland": "savanna grassland landscape wildlife habitat",
    "Forest_Ecosystem": "dense forest ecosystem tropical jungle trees",
    "Desert_SemiArid": "desert semi-arid sand dunes landscape",
    "Mountain_Alpine": "mountain alpine snow peaks scenic landscape",
    "Marine_Ocean": "ocean sea marine ecosystem coral reef coastal water",
    "Freshwater_Ecosystem": "river lake pond freshwater wetland ecosystem",
    "Human_Settlements": "urban rural human settlements farms cities villages",
    "Wildlife_Reserve": "wildlife reserve national park natural landscape",
    "Arctic_Polar": "arctic polar snowy ice landscape tundra wildlife"
}

DATASET_DIR = "./scene_dataset"
IMAGES_PER_CLASS = 1500
MIN_SIZE = 400
SPLIT_RATIOS = (0.7, 0.15, 0.15)
HEADERS_PEXELS = {"Authorization": PEXELS_KEY}

# =========================
# HELPER FUNCTIONS
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def valid_image(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w >= MIN_SIZE and h >= MIN_SIZE
    except Exception:
        return False

# =========================
# API FETCHERS
# =========================
def fetch_unsplash_images(query, per_page=30, pages=10):
    urls = []
    base_url = "https://api.unsplash.com/search/photos"
    for page in range(1, pages + 1):
        params = {"query": query, "per_page": per_page, "page": page}
        r = requests.get(base_url, params=params, headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"})
        if r.status_code == 200:
            data = r.json().get("results", [])
            urls += [img["urls"]["full"] for img in data if "urls" in img]
        time.sleep(1)
    return urls

def fetch_pexels_images(query, per_page=80, pages=10):
    urls = []
    base_url = "https://api.pexels.com/v1/search"
    for page in range(1, pages + 1):
        params = {"query": query, "per_page": per_page, "page": page}
        r = requests.get(base_url, params=params, headers=HEADERS_PEXELS)
        if r.status_code == 200:
            data = r.json().get("photos", [])
            urls += [img["src"]["original"] for img in data if "src" in img]
        time.sleep(0.5)
    return urls

# =========================
# MAIN DOWNLOAD LOGIC
# =========================
def download_for_label(label, query):
    folder = os.path.join(DATASET_DIR, label)
    ensure_dir(folder)
    seen_hashes = set()
    downloaded = 0

    print(f"\n🔍 Fetching {label} images...")

    unsplash_urls = fetch_unsplash_images(query)
    pexels_urls = fetch_pexels_images(query)
    all_urls = list(dict.fromkeys(unsplash_urls + pexels_urls))  # dedupe by URL

    random.shuffle(all_urls)
    print(f"Found {len(all_urls)} candidate URLs for {label}")

    for url in all_urls:
        save_path = os.path.join(folder, f"{downloaded:04d}.jpg")
        if download_image(url, save_path) and valid_image(save_path):
            try:
                with Image.open(save_path) as im:
                    ph = imagehash.average_hash(im)
                if ph in seen_hashes:
                    os.remove(save_path)
                    continue
                seen_hashes.add(ph)
                downloaded += 1
                print(f"{label}: {downloaded}/{IMAGES_PER_CLASS}", end="\r")
            except Exception:
                os.remove(save_path)
                continue
        if downloaded >= IMAGES_PER_CLASS:
            break
        time.sleep(random.uniform(0.1, 0.3))

    print(f"\n✅ {label} done ({downloaded} images)")

# =========================
# SPLIT TRAIN/VAL/TEST
# =========================
def split_dataset():
    print("\n📂 Splitting into train/val/test ...")
    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue

        files = [f for f in os.listdir(label_path) if f.endswith(".jpg")]
        random.shuffle(files)
        n = len(files)
        n_train = int(SPLIT_RATIOS[0] * n)
        n_val = int(SPLIT_RATIOS[1] * n)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }

        for split, file_list in splits.items():
            split_dir = os.path.join(DATASET_DIR, split, label)
            ensure_dir(split_dir)
            for f in file_list:
                shutil.move(os.path.join(label_path, f), os.path.join(split_dir, f))
        shutil.rmtree(label_path)
    print("✅ Split complete (70/15/15)")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_dir(DATASET_DIR)
    for label, query in SCENE_LABELS.items():
        download_for_label(label, query)
    split_dataset()
    print("\n🎉 Dataset downloaded, cleaned, and split successfully!")
