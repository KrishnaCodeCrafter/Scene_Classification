import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings

# ======================================================
# BASIC SETUP
# ======================================================
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = "./scene_dataset_final"        # cleaned dataset from Phase 2
MODEL_NAME = "efficientnet_b3"            # can switch to "efficientnet_b0" if VRAM tight
BATCH_SIZE = 32                           # starting batch size (auto-adjusts)
EPOCHS = 20
LR = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
if device == "cuda":
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"💾 CUDA Version: {torch.version.cuda}")
else:
    print("⚠️ GPU not detected — running on CPU (slow)")

# ======================================================
# DATA TRANSFORMS
# ======================================================
train_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================================================
# TRAINING FUNCTION
# ======================================================
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, writer):
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Training loop with AMP
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            try:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠️ OOM encountered. Reducing batch size automatically...")
                    torch.cuda.empty_cache()
                    return "OOM"
                else:
                    raise e

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        val_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}: Val Acc = {acc:.2f} %, Loss = {val_loss:.4f}")

        # Log metrics
        writer.add_scalar("Loss/train", val_loss, epoch)
        writer.add_scalar("Accuracy/val", acc, epoch)
        writer.flush()

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"best_{MODEL_NAME}_scene.pth")
            print(f"💾 Model improved — saved best_{MODEL_NAME}_scene.pth")

        torch.cuda.empty_cache()

    print(f"\n🎯 Training complete — best validation accuracy: {best_acc:.2f} %")
    return "DONE"

# ======================================================
# MAIN (Windows-safe)
# ======================================================
if __name__ == "__main__":
    # ---- Dataset Setup ----
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_tf)
    NUM_CLASSES = len(train_ds.classes)

    # ---- Data Loaders ----
    def create_loaders(batch_size):
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)
        )

    train_loader, val_loader, test_loader = create_loaders(BATCH_SIZE)

    # ---- Model ----
    print(f"🔧 Loading {MODEL_NAME} pretrained on ImageNet...")
    model = getattr(models, MODEL_NAME)(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    # ---- Optimizer, Loss, Scheduler ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---- TensorBoard Logger ----
    writer = SummaryWriter("runs/scene_training")

    # ---- Automatic OOM Recovery ----
    while True:
        result = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, writer)
        if result == "DONE":
            break
        elif result == "OOM":
            BATCH_SIZE = max(BATCH_SIZE // 2, 4)
            print(f"🔁 Retrying with smaller batch size: {BATCH_SIZE}")
            train_loader, val_loader, test_loader = create_loaders(BATCH_SIZE)
            torch.cuda.empty_cache()
        else:
            break

    writer.close()
