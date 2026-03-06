#!/usr/bin/env python3
"""
main.py
Full pipeline orchestrator for:
- frame extraction
- face extraction
- rPPG extraction + HR quality
- CNN per-frame deepfake scoring
- Transformer temporal deepfake scoring
- Decision fusion
"""

import argparse
import sys
import torch
import numpy as np
from tqdm import tqdm

# Local imports (make sure src is on your PYTHONPATH or run from project root)
from src.preprocessing.frame_generator import load_video_frames
from src.preprocessing.face_extraction import extract_face
from src.rppg.rppg_extractor import extract_rppg
from src.rppg.heart_rate_estimator import estimate_hr_quality
from src.deepfake.cnn_detector import load_cnn_model, predict_frame
from src.deepfake.transformer_detector import load_transformer_model, DeepfakeTransformer
from src.fusion.decision_fusion import fuse_decisions
from src.micro_expression.micro_expression_detector import analyze_micro_expressions


def run_pipeline(video_path,
                 max_frames=None,
                 face_size=(128, 128),
                 use_transformer=True,
                 transformer_frames=16,
                 device=None,
                 cnn_model_path=None,
                 transformer_model_path=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # 1) Load frames (keep original size for face detector)
    print("Loading video frames...")
    frames = load_video_frames(video_path, max_frames=max_frames)
    if len(frames) == 0:
        print("No frames loaded. Exiting.")
        return

    # 2) Extract face crops
    print("Extracting faces from frames...")
    face_frames = []
    for f in tqdm(frames, desc="Faces"):
        face = extract_face(f, target_size=face_size)
        if face is not None:
            face_frames.append(face)
    if len(face_frames) == 0:
        print("No faces detected in any frame. Exiting.")
        return

    print(f"Detected {len(face_frames)} face frames (after filtering).")

    # 3) rPPG extraction
    print("Extracting rPPG signal (green-channel) and estimating HR...")
    hr_bpm, raw_signal = extract_rppg(face_frames)
    if hr_bpm is None:
        print("rPPG extraction failed or not enough frames.")
        hr_bpm = None
        hr_quality_score = 0.0
    else:
        hr_quality_score = estimate_hr_quality(raw_signal, hr_bpm)
    print(f"Estimated HR: {hr_bpm} BPM, HR quality score: {hr_quality_score:.3f}")

    # 4) Load CNN model (frame-level)
    print("Loading CNN model...")
    cnn_model = load_cnn_model(model_path=cnn_model_path, device=device)
    cnn_model.eval()

    #5) Micro-expression analysis (optional, can be integrated into fusion later)
    print("Analyzing micro-expressions...")
    micro_expression_score = analyze_micro_expressions(face_frames)
    print("Micro-expression score:", micro_expression_score)

    # Run CNN on each face and average scores
    print("Running CNN on frames...")
    cnn_scores = []
    for face in tqdm(face_frames, desc="CNN"):
        try:
            score = predict_frame(cnn_model, face)
            cnn_scores.append(score)
        except Exception as e:
            # In case of GPU mismatch or other issue, print and continue
            print(f"Warning: CNN prediction failed on a frame: {e}", file=sys.stderr)

    if len(cnn_scores) == 0:
        avg_cnn_score = 0.0
    else:
        avg_cnn_score = float(np.mean(cnn_scores))

    print(f"Average CNN fake probability: {avg_cnn_score:.3f}")

    #6) Transformer temporal model (optional). We'll prepare a tensor and call model directly.
    avg_transformer_score = None
    if use_transformer:
        print("Loading Transformer model...")
        transformer_model = load_transformer_model(model_path=transformer_model_path, device=device)
        transformer_model.eval()

        # Choose up to `transformer_frames` equally spaced frames
        T = min(len(face_frames), transformer_frames)
        if T == 0:
            avg_transformer_score = None
            print("Not enough frames for transformer.")
        else:
            idxs = np.linspace(0, len(face_frames)-1, T, dtype=int)
            selected = [face_frames[i] for i in idxs]

            # Convert to tensor: shape (B=1, T, C, H, W)
            tensor = torch.tensor(selected, dtype=torch.float32).permute(0,3,1,2).unsqueeze(0)
            # permute produced (T,3,H,W) then unsqueeze -> (1,T,3,H,W)

            tensor = tensor.to(device)
            with torch.no_grad():
                try:
                    pred = transformer_model(tensor)  # returns (B,1)
                    avg_transformer_score = float(pred.cpu().numpy().squeeze())
                except Exception as e:
                    print(f"Warning: Transformer inference failed: {e}", file=sys.stderr)
                    avg_transformer_score = None

        if avg_transformer_score is not None:
            print(f"Transformer (video-level) fake probability: {avg_transformer_score:.3f}")
        else:
            print("Transformer score unavailable.")

    # 7) Final fusion
    # Use CNN score primarily; if transformer exists, combine CNN+Transformer into a single cnn_score proxy
    final_cnn_score_for_fusion = avg_cnn_score
    if avg_transformer_score is not None:
        # weight transformer more for temporal evidence
        final_cnn_score_for_fusion = 0.5 * avg_cnn_score + 0.5 * avg_transformer_score
        print(f"Combined visual score (CNN+Transformer): {final_cnn_score_for_fusion:.3f}")

    fusion_result = fuse_decisions(
    final_cnn_score_for_fusion,
    hr_quality_score,
    micro_expression_score
    )
    print("\n===== FINAL RESULT =====")
    print(f"Final label : {fusion_result['final_label']}")
    print(f"CNN score   : {fusion_result['cnn_score']:.3f}")
    print(f"HR quality  : {fusion_result['hr_quality_score']:.3f}")
    print(f"MicroExp    : {fusion_result['micro_expression_score']:.3f}")
    print(f"Confidence  : {fusion_result['confidence']:.3f}")
    print("========================")

    # Optionally return structured output
    return {
        "video_path": video_path,
        "hr_bpm": hr_bpm,
        "hr_quality": hr_quality_score,
        "cnn_score": avg_cnn_score,
        "transformer_score": avg_transformer_score,
        "fusion": fusion_result
    }

def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Detection Pipeline (rPPG + Visual Models)")
    p.add_argument("--video", type=str, required=False, default="data/samples/sample1.mp4", help="Path to input video")
    p.add_argument("--max_frames", type=int, default=None, help="Limit number of frames to process (for testing)")
    p.add_argument("--no_transformer", action="store_true", help="Disable transformer model inference")
    p.add_argument("--cnn_model", type=str, default=None, help="Path to a saved CNN model (.pt)")
    p.add_argument("--transformer_model", type=str, default=None, help="Path to a saved transformer model (.pt)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_pipeline(
        video_path=args.video,
        max_frames=args.max_frames,
        use_transformer=(not args.no_transformer),
        device=device,
        cnn_model_path=args.cnn_model,
        transformer_model_path=args.transformer_model
    )
