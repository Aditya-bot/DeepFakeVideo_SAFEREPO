import os
import shutil
from collections import defaultdict

# -------- PROJECT ROOT FIX --------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def group_by_video(files, chunk_size=30):
    groups = {}
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    for i in range(0, len(files), chunk_size):
        vid = f"video_{i//chunk_size:04d}"
        groups[vid] = files[i:i+chunk_size]

    return groups


def prepare_dataset(input_root, output_root):

    for label in ["real", "fake"]:
        input_path = os.path.join(input_root, label)
        output_path = os.path.join(output_root, label)

        os.makedirs(output_path, exist_ok=True)

        files = sorted(os.listdir(input_path))
        grouped = group_by_video(files)

        print(f"\nProcessing {label}...")

        count = 0

        for vid, frames in grouped.items():

            if len(frames) < 5:
                continue

            video_folder = os.path.join(output_path, vid)
            os.makedirs(video_folder, exist_ok=True)

            frames = sorted(frames)

            for i, f in enumerate(frames):
                src = os.path.join(input_path, f)
                dst = os.path.join(video_folder, f"{i:04d}.jpg")

                shutil.copy(src, dst)

            count += 1

        print(f"Created {count} video folders for {label}")


if __name__ == "__main__":
    prepare_dataset(
        input_root=os.path.join(PROJECT_ROOT, "data", "dataset_faces"),
        output_root=os.path.join(PROJECT_ROOT, "dataset_micro")
    )