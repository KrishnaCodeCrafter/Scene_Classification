import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_PATH = "./best_efficientnet_b3_scene.pth"
MODEL_NAME = "efficientnet_b3"
DATA_DIR = "./scene_dataset_final"  # just for class names reference
IMG_PATH = "test/test_2.jpg"         # path to your input image
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# TRANSFORMS (must match training transforms)
# ======================================================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================================================
# LOAD CLASS NAMES
# ======================================================
from torchvision import datasets
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
class_names = test_ds.classes

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
print("✅ Model loaded successfully.")

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

    print(f"🧠 Prediction: {predicted_label} ({confidence_value:.2f}% confidence)")

    # --- Optional: visualize the prediction ---
    plt.imshow(image)
    plt.title(f"{predicted_label} ({confidence_value:.2f}%)")
    plt.axis("off")
    plt.show()

    return predicted_label, confidence_value

# ======================================================
# RUN PREDICTION
# ======================================================
if __name__ == "__main__":
    if not os.path.exists(IMG_PATH):
        print(f"⚠️ Image not found: {IMG_PATH}")
    else:
        predict_image(IMG_PATH)
