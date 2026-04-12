import os
import torch
import numpy as np
import cv2

from .micro_cnn import MicroExpressionResNet

# -------- PROJECT ROOT --------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- LOAD MODEL --------
model = MicroExpressionResNet().to(device)

model_path = os.path.join(PROJECT_ROOT, "models", "micro_cnn.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def predict_micro_expression(frames):
    if len(frames) < 2:
        return 0.5

    scores = []

    for i in range(1, len(frames)):
        f1 = frames[i - 1]
        f2 = frames[i]

        # normalize if needed
        if f1.max() > 1.0:
            f1 = f1.astype("float32") / 255.0
        if f2.max() > 1.0:
            f2 = f2.astype("float32") / 255.0

        # motion
        diff = np.abs(f2 - f1) * 5.0
        diff = np.clip(diff, 0, 1)

        diff = cv2.resize(diff, (224, 224))
        diff = diff.transpose(2, 0, 1)

        tensor = torch.tensor(diff, dtype=torch.float32)\
            .unsqueeze(0).to(device)

        with torch.no_grad():
            score = model(tensor).item()
            scores.append(score)

        raw_score = float(np.mean(scores))

        # -------- CALIBRATION --------
        # push fake lower, keep real high
        calibrated = (raw_score - 0.4) / (1 - 0.4)
        calibrated = np.clip(calibrated, 0, 1)
    
    return float(calibrated)