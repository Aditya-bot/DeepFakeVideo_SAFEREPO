import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from main import run_pipeline

# ================= CONFIG =================
DATASET_PATH = "data/test_videos"
USE_TRANSFORMER = False   # set True if using transformer
CLASS_NAMES = ["real", "fake"]

# ================= STORAGE =================
y_true = []
y_pred = []

# ================= EVALUATION LOOP =================
for label_name in CLASS_NAMES:
    folder = os.path.join(DATASET_PATH, label_name)

    if not os.path.exists(folder):
        print(f"Warning: Folder not found -> {folder}")
        continue

    for video_file in os.listdir(folder):
        video_path = os.path.join(folder, video_file)

        print(f"Testing: {video_file}")

        try:
            result = run_pipeline(
                video_path=video_path,
                use_transformer=USE_TRANSFORMER
            )

            pred_label = result["final_label"].lower()

            y_true.append(label_name)
            y_pred.append(pred_label)

            print(f"Predicted: {pred_label} | Actual: {label_name}")

        except Exception as e:
            print(f"Error processing {video_file}: {e}")

# ================= METRICS =================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred,
    pos_label="fake",
    average="binary"
)
recall = recall_score(
    y_true, y_pred,
    pos_label="fake",
    average="binary"
)
f1 = f1_score(
    y_true, y_pred,
    pos_label="fake",
    average="binary"
)

cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)
sensitivity = recall   # same as recall

# ================= PRINT RESULTS =================
print("\n" + "="*50)
print("FINAL MULTIMODAL SYSTEM EVALUATION")
print("="*50)

print(f"Total Samples : {len(y_true)}")
print(f"Accuracy      : {accuracy*100:.2f}%")
print(f"Precision     : {precision*100:.2f}%")
print(f"Recall        : {recall*100:.2f}%")
print(f"F1 Score      : {f1*100:.2f}%")
print(f"Sensitivity   : {sensitivity*100:.2f}%")
print(f"Specificity   : {specificity*100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred))

# ================= SAVE RESULTS =================
with open("evaluation_results.txt", "w") as f:
    f.write("FINAL MULTIMODAL SYSTEM EVALUATION\n")
    f.write("="*50 + "\n")
    f.write(f"Total Samples : {len(y_true)}\n")
    f.write(f"Accuracy      : {accuracy*100:.2f}%\n")
    f.write(f"Precision     : {precision*100:.2f}%\n")
    f.write(f"Recall        : {recall*100:.2f}%\n")
    f.write(f"F1 Score      : {f1*100:.2f}%\n")
    f.write(f"Sensitivity   : {sensitivity*100:.2f}%\n")
    f.write(f"Specificity   : {specificity*100:.2f}%\n")
    f.write(f"\nConfusion Matrix:\n{cm}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_true, y_pred))