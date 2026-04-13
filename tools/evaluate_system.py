import os
from main import run_pipeline

DATASET_PATH = "data/test_videos"

correct = 0
total = 0

for label in ["real", "fake"]:
    folder = os.path.join(DATASET_PATH, label)

    for video in os.listdir(folder):
        video_path = os.path.join(folder, video)

        result = run_pipeline(
            video_path=video_path,
            use_transformer=False
        )

        pred = result["final_label"].lower()

        print(f"{video} -> Pred: {pred} | Actual: {label}")

        if pred == label:
            correct += 1

        total += 1

accuracy = correct / total

print("\n===== FINAL SYSTEM ACCURACY =====")
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Accuracy: {accuracy * 100:.2f}%")