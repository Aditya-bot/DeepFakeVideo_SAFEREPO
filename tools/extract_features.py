import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

# ================= PATH FIX =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.deepfake.cnn_detector import load_cnn_model

# ================= CONFIG =================
DATASET_PATH = "data/ff_dataset"   # <-- your dataset path
SAVE_PATH = "data/features"
SEQ_LEN = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD MODEL =================
model = load_cnn_model("models/cnn_deepfake.pt", device=device)
model.eval()


# ================= PREPROCESS =================
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        frame[:, :, i] = (frame[:, :, i] - mean[i]) / std[i]

    frame = torch.tensor(frame, dtype=torch.float32)\
        .permute(2, 0, 1)\
        .unsqueeze(0)

    return frame.to(device)


# ================= FEATURE EXTRACTION =================
def extract_video_features(video_path, seq_len=SEQ_LEN):
    cap = cv2.VideoCapture(video_path)

    features = []

    while len(features) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)

        with torch.no_grad():
            feat = model(input_tensor, return_features=True)

        features.append(feat.squeeze().cpu().numpy())

    cap.release()

    if len(features) < seq_len:
        return None

    return np.array(features)  # shape: (seq_len, 512)


# ================= MAIN PROCESS =================
def process_dataset(dataset_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    fake_classes = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]

    real_classes = ["original"]

    # -------- FAKE --------
    for cls in fake_classes:
        class_path = os.path.join(dataset_path, cls)

        for video in tqdm(os.listdir(class_path), desc=f"Processing {cls}"):

            video_path = os.path.join(class_path, video)

            if not video_path.endswith((".mp4", ".avi", ".mov")):
                continue

            features = extract_video_features(video_path)

            if features is None:
                continue

            save_file = os.path.join(save_path, f"{cls}_{video}.npy")

            np.save(save_file, {
                "features": features,
                "label": 0  # FAKE
            })

    # -------- REAL --------
    for cls in real_classes:
        class_path = os.path.join(dataset_path, cls)

        for video in tqdm(os.listdir(class_path), desc=f"Processing {cls}"):

            video_path = os.path.join(class_path, video)

            if not video_path.endswith((".mp4", ".avi", ".mov")):
                continue

            features = extract_video_features(video_path)

            if features is None:
                continue

            save_file = os.path.join(save_path, f"{cls}_{video}.npy")

            np.save(save_file, {
                "features": features,
                "label": 1  # REAL
            })


# ================= RUN =================
if __name__ == "__main__":
    process_dataset(DATASET_PATH, SAVE_PATH)