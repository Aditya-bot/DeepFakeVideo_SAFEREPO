import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.deepfake.transformer_model import DeepfakeTransformer
from src.deepfake.sequence_dataset import SequenceDataset

# ================= CONFIG =================
FEATURE_DIR = "data/features"
BATCH_SIZE = 16
EPOCHS = 10
LR = 5e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= DATA =================
dataset = SequenceDataset(FEATURE_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================= MODEL =================
model = DeepfakeTransformer().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# ================= TRAIN =================
for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val   Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/transformer.pt")
        print("✅ Best transformer saved!")

print("\nTraining complete!")
print("Best Val Acc:", best_val_acc)