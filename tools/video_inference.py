import os
import sys
import cv2
import torch
import numpy as np

# Fix path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.deepfake.cnn_detector import load_cnn_model


# ================= LOAD MODEL =================
model = load_cnn_model("models/cnn_deepfake.pt")
model.eval()

device = next(model.parameters()).device


# ================= PREPROCESS =================
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        frame[:, :, i] = (frame[:, :, i] - mean[i]) / std[i]

    frame = torch.tensor(frame, dtype=torch.float32)\
        .permute(2, 0, 1)\
        .unsqueeze(0)

    return frame.to(device)


# ================= VIDEO PREDICTION =================
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_preds = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every 5th frame (faster)
        if frame_count % 5 != 0:
            frame_count += 1
            continue

        input_tensor = preprocess(frame)

        with torch.no_grad():
            output = model(input_tensor)
            prob_real = torch.sigmoid(output).item()

        frame_preds.append(prob_real)

        frame_count += 1

    cap.release()

    if len(frame_preds) == 0:
        return "No frames processed"

    avg_prob = np.mean(frame_preds)

    print(f"Average REAL probability: {avg_prob:.4f}")

    # FINAL DECISION
    if avg_prob > 0.5:
        return "REAL"
    else:
        return "FAKE"


# ================= RUN =================
if __name__ == "__main__":

    video_path = "C:\\Users\\write\\OneDrive\\Documents\\projects\\FINAL_YEAR_PROJECT\\DeepFakeVideo_DetectionModel\\DeepFakeVideo_SAFEREPO\\data\\samples\\test.mp4"  # change this

    result = predict_video(video_path)

    print("Final Prediction:", result)