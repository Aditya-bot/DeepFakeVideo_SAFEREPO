import os
import sys
import torch
from torch.utils.data import DataLoader

# ---- PROJECT ROOT ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.micro_expression.micro_cnn import MicroExpressionResNet
from src.micro_expression.motion_dataset import MotionDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MotionDataset(os.path.join(PROJECT_ROOT, "dataset_micro"))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MicroExpressionResNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

print(f"Training on {len(dataset)} samples...")

for epoch in range(10):
    total_loss = 0
    total_acc = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        x = torch.clamp(x, 0, 1)

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds_binary = (preds > 0.5).float()
        acc = (preds_binary == y).float().mean()
        total_acc += acc.item()

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)

    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.4f}")

# ---- SAVE ----
torch.save(
    model.state_dict(),
    os.path.join(PROJECT_ROOT, "models", "micro_cnn.pt")
)

print("Model saved ✅")