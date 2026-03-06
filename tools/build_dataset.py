import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.frame_generator import load_video_frames
from src.preprocessing.face_extraction import extract_face


DATASET_PATH = "data/ff_dataset"
OUTPUT_PATH = "data/dataset_faces"

REAL_FOLDER = "original"

FAKE_FOLDERS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]

FACE_SIZE = (128, 128)

FRAMES_PER_VIDEO = 10


def ensure_uint8(img):
    """
    Ensures image is uint8 before saving.
    Handles both normalized (0–1) and uint8 (0–255) images.
    """

    if img.dtype == np.float32 or img.dtype == np.float64:

        if img.max() <= 1.0:
            img = img * 255.0

        img = img.astype(np.uint8)

    return img


def save_faces_from_video(video_path, label, counter):

    frames = load_video_frames(video_path, max_frames=FRAMES_PER_VIDEO)

    for frame in frames:

        face = extract_face(frame, target_size=FACE_SIZE)

        if face is None:
            continue

        face = ensure_uint8(face)

        save_path = os.path.join(
            OUTPUT_PATH,
            label,
            f"{counter}.jpg"
        )

        cv2.imwrite(save_path, face)

        counter += 1

    return counter


def process_folder(folder_name, label):

    folder_path = os.path.join(DATASET_PATH, folder_name)

    videos = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".mp4")
    ]

    counter = 0

    for video in tqdm(videos, desc=f"Processing {folder_name}"):

        counter = save_faces_from_video(video, label, counter)


def main():

    os.makedirs(os.path.join(OUTPUT_PATH, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "fake"), exist_ok=True)

    print("\nProcessing REAL videos...\n")
    process_folder(REAL_FOLDER, "real")

    print("\nProcessing FAKE videos...\n")

    for folder in FAKE_FOLDERS:
        process_folder(folder, "fake")


if __name__ == "__main__":
    main()