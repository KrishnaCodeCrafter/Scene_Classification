import os
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# CONFIGURATION
# ======================================================
DATA_DIR = "./scene_dataset_final"
MODEL_PATH = "./best_efficientnet_b3_scene.pth"
MODEL_NAME = "efficientnet_b3"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# DATA TRANSFORM
# ======================================================
test_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================================================
# LOAD DATASET
# ======================================================
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
class_names = test_ds.classes
print(f"🧾 Classes found: {class_names}")

# ======================================================
# LOAD MODEL
# ======================================================
print(f"🔧 Loading {MODEL_NAME} model...")
model = getattr(models, MODEL_NAME)(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ======================================================
# EVALUATION
# ======================================================
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ======================================================
# METRICS
# ======================================================
print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

cm = confusion_matrix(all_labels, all_preds)
acc = (np.trace(cm) / np.sum(cm)) * 100
print(f"🎯 Overall Test Accuracy: {acc:.2f}%")

# ======================================================
# PLOT CONFUSION MATRIX
# ======================================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix ({MODEL_NAME}) — Test Accuracy: {acc:.2f}%")
plt.tight_layout()
plt.show()
