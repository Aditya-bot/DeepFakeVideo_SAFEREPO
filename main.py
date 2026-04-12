#!/usr/bin/env python3

import argparse
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================= IMPORTS =================
from src.preprocessing.frame_generator import load_video_frames
from src.preprocessing.face_extraction import extract_face
from src.rppg.rppg_extractor import extract_rppg
from src.rppg.heart_rate_estimator import estimate_hr_quality
from src.deepfake.cnn_detector import load_cnn_model, predict_frame
from src.deepfake.transformer_detector import load_transformer_model
from src.fusion.decision_fusion import fuse_predictions as fuse_decisions
#from src.micro_expression.micro_expression_detector import analyze_micro_expressions
from src.micro_expression.micro_inference import predict_micro_expression


# ================= MAIN PIPELINE =================
def run_pipeline(
    video_path,
    max_frames=None,
    face_size=(128, 128),
    use_transformer=True,
    transformer_frames=16,
    device=None,
    cnn_model_path="models/cnn_deepfake.pt",
    transformer_model_path="models/transformer.pt"
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # ================= 1. LOAD FRAMES =================
    print("\nLoading video frames...")
    frames = load_video_frames(video_path, max_frames=max_frames)

    if len(frames) == 0:
        print("❌ No frames loaded.")
        return

    # ================= 2. FACE EXTRACTION =================
    print("Extracting faces...")
    face_frames = []

    for f in tqdm(frames, desc="Faces"):
        face = extract_face(f, target_size=face_size)
        if face is not None:
            face_frames.append(face)

    print(f"Frames: {len(frames)} | Faces: {len(face_frames)}")

    if len(face_frames) == 0:
        print("❌ No faces detected.")
        return

    # ================= 3. rPPG =================
    print("\nExtracting rPPG...")

    # FIX: Get FPS
    cap = cv2.VideoCapture(video_path)
    fs = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fs == 0:
        fs = 30  # fallback

    hr_bpm, raw_signal = extract_rppg(face_frames, fs)

    # FIX: safe handling
    if hr_bpm is None or raw_signal is None:
        hr_quality_score = 0.0
    else:
        hr_quality_score = estimate_hr_quality(raw_signal, hr_bpm, fs)

    # Better print
    if hr_bpm is not None:
        print(f"HR: {hr_bpm:.2f} BPM | Quality: {hr_quality_score:.3f}")
    else:
        print(f"HR: Not detected | Quality: {hr_quality_score:.3f}")

    # ================= 4. LOAD CNN =================
    print("\nLoading CNN model...")
    cnn_model = load_cnn_model(model_path=cnn_model_path, device=device)
    cnn_model.eval()
    print(f"Loaded CNN from: {cnn_model_path}")

    # ================= 5. MICRO-EXPRESSION =================
    print("\nAnalyzing micro-expressions...")
    micro_expression_score = predict_micro_expression(face_frames)
    print(f"Micro-expression score: {micro_expression_score:.3f}")

    # ================= 6. CNN INFERENCE =================
    print("\nRunning CNN...")
    cnn_scores = []

    for face in tqdm(face_frames, desc="CNN"):
        try:
            score = predict_frame(cnn_model, face)
            cnn_scores.append(score)
        except Exception as e:
            print(f"Warning: CNN failed on a frame: {e}", file=sys.stderr)

    avg_cnn_score = float(np.mean(cnn_scores)) if cnn_scores else 0.5
    print(f"Average CNN score: {avg_cnn_score:.3f}")

    # ================= 7. TRANSFORMER =================
    avg_transformer_score = None

    if use_transformer:
        print("\nLoading Transformer model...")
        transformer_model = load_transformer_model(
            model_path=transformer_model_path,
            device=device
        )
        transformer_model.eval()
        print(f"Loaded Transformer from: {transformer_model_path}")

        T = min(len(face_frames), transformer_frames)

        if T > 0:
            idxs = np.linspace(0, len(face_frames) - 1, T, dtype=int)
            selected = [face_frames[i] for i in idxs]

            tensor = torch.tensor(selected, dtype=torch.float32)\
                .permute(0, 3, 1, 2)\
                .unsqueeze(0)\
                .to(device)

            with torch.no_grad():
                try:
                    pred = transformer_model(tensor)
                    avg_transformer_score = float(pred.cpu().numpy().squeeze())
                except Exception as e:
                    print(f"Transformer error: {e}", file=sys.stderr)

        if avg_transformer_score is not None:
            print(f"Transformer score: {avg_transformer_score:.3f}")
        else:
            print("Transformer unavailable")

    # ================= 8. FUSION =================
    final_visual_score = avg_cnn_score

    if avg_transformer_score is not None:
        final_visual_score = 0.5 * avg_cnn_score + 0.5 * avg_transformer_score
        print(f"Combined visual score: {final_visual_score:.3f}")

    fusion_result = fuse_decisions(
        final_visual_score,
        hr_quality_score,
        micro_expression_score,
        hr_bpm  # FIX: pass HR BPM
    )

    # ================= FINAL OUTPUT =================
    print("\n===== FINAL RESULT =====")
    print(f"Final label : {fusion_result['final_label']}")
    print(f"CNN score   : {fusion_result['cnn_score']:.3f}")
    print(f"HR quality  : {fusion_result['hr_quality_score']:.3f}")
    print(f"MicroExp    : {fusion_result['micro_expression_score']:.3f}")
    print(f"Confidence  : {fusion_result['confidence']:.3f}")
    print("========================")

    return fusion_result


# ================= ARGUMENTS =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/samples/test.mp4")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_transformer", action="store_true")
    return parser.parse_args()


# ================= RUN =================
if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_pipeline(
        video_path=args.video,
        max_frames=args.max_frames,
        use_transformer=(not args.no_transformer),
        device=device
    )