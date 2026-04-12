import os
import sys
import cv2
# -------- PROJECT ROOT --------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add BOTH root and src explicitly
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

print("Project root:", PROJECT_ROOT)

from micro_expression.micro_inference import predict_micro_expression


def load_frames(folder, max_frames=30):
    frames = []

    files = sorted(os.listdir(folder), key=lambda x: int(os.path.splitext(x)[0]))

    for f in files[:max_frames]:
        path = os.path.join(folder, f)
        img = cv2.imread(path)

        if img is not None:
            frames.append(img)

    return frames


# -------- TEST REAL --------
real_root = os.path.join(PROJECT_ROOT, "dataset_micro", "real")
real_video = os.listdir(real_root)[0]

real_frames = load_frames(os.path.join(real_root, real_video))
real_score = predict_micro_expression(real_frames)

print(f"REAL score: {real_score:.4f}")


# -------- TEST FAKE --------
fake_root = os.path.join(PROJECT_ROOT, "dataset_micro", "fake")
fake_video = os.listdir(fake_root)[0]

fake_frames = load_frames(os.path.join(fake_root, fake_video))
fake_score = predict_micro_expression(fake_frames)

print(f"FAKE score: {fake_score:.4f}")