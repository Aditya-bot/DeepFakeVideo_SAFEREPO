import cv2

def load_video_frames(video_path, max_frames=None, resize=None):
    """
    Loads frames from a video file.

    Args:
        video_path (str): path to the input video
        max_frames (int or None): limit on number of frames (optional)
        resize (tuple or None): (width, height) to resize frames

    Returns:
        list of numpy arrays (frames)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return []

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optional resize
        if resize is not None:
            frame = cv2.resize(frame, resize)

        frames.append(frame)
        count += 1

        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    return frames
