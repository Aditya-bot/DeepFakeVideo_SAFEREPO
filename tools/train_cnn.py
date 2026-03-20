import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

print("Script started")

# ================= PATH FIX =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.deepfake.cnn_detector import load_cnn_model

# ================= CONFIG =================
DATASET_PATH = "data/dataset_faces"
MODEL_SAVE_PATH = "models/cnn_deepfake.pt"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 128


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ================= MODEL =================
    model = load_cnn_model(model_path=None, device=device)
    model.train()

    # ================= LOSS =================
    # Use BCEWithLogitsLoss if NO sigmoid in model
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ================= TRANSFORMS =================
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

    # ================= DATA =================
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Classes:", dataset.classes)

    best_val_acc = 0.0

    # ================= TRAIN LOOP =================
    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_acc = correct / total

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # -------- SAVE BEST MODEL --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Best model saved!")

    print("\nTraining complete!")
    print("Best Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    train()
