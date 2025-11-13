import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_PATH = "./best_efficientnet_b3_scene.pth"  # trained model
MODEL_NAME = "efficientnet_b3"
CLASS_JSON = "./class_names.json"                # JSON file containing class names
TEST_DIR = "./test"                              # folder with test images
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# LOAD CLASS NAMES FROM JSON
# ======================================================
if not os.path.exists(CLASS_JSON):
    raise FileNotFoundError(f"❌ Could not find {CLASS_JSON}. Please create it first.")

with open(CLASS_JSON, "r") as f:
    data = json.load(f)
class_names = data.get("class_names", [])
if not class_names:
    raise ValueError("❌ No class names found in class_names.json")

print(f"✅ Loaded {len(class_names)} classes from {CLASS_JSON}: {class_names}\n")

# ======================================================
# IMAGE TRANSFORMS (same as training)
# ======================================================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================================================
# LOAD MODEL
# ======================================================
print(f"🔧 Loading model: {MODEL_NAME}")
model = getattr(models, MODEL_NAME)(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully.\n")

# ======================================================
# PREDICT FUNCTION
# ======================================================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    
    predicted_label = class_names[pred_class.item()]
    confidence_value = confidence.item() * 100
    return predicted_label, confidence_value

# ======================================================
# RUN PREDICTIONS FOR ALL IMAGES IN TEST FOLDER
# ======================================================
if not os.path.exists(TEST_DIR):
    print(f"⚠️ Test folder not found: {TEST_DIR}")
else:
    image_files = [f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("⚠️ No images found in test folder.")
    else:
        print(f"📁 Found {len(image_files)} images in '{TEST_DIR}'\n")
        for img_file in image_files:
            img_path = os.path.join(TEST_DIR, img_file)
            label, conf = predict_image(img_path)

            print(f"🖼️ {img_file} → {label} ({conf:.2f}% confidence)")

            # Display image with label
            image = Image.open(img_path)
            plt.imshow(image)
            plt.title(f"{label} ({conf:.2f}%)")
            plt.axis("off")
            plt.show()
