import os
import sys
import cv2
import torch

# Fix path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.deepfake.cnn_detector import load_cnn_model, predict_frame

# ================= LOAD MODEL =================
model = load_cnn_model("models/cnn_deepfake.pt")

print("Model loaded successfully!")

# ================= TEST IMAGE =================
image_path = "C:\\Users\\write\\OneDrive\\Documents\\projects\\FINAL_YEAR_PROJECT\\DeepFakeVideo_DetectionModel\\DeepFakeVideo_SAFEREPO\\data\\samples\\12.jpg"  # change this

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # IMPORTANT
image = cv2.resize(image, (224, 224))
image = image / 255.0

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

for i in range(3):
    image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
# ================= PREDICTION =================
prob = predict_frame(model, image)

print("Fake probability:", prob)

if prob < 0.5:
    print("Prediction: FAKE")
else:
    print("Prediction: REAL")