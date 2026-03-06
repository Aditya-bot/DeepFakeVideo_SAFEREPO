import cv2
import os

# -------------------------------------------------------------
# 1. Get video metadata
# -------------------------------------------------------------
def get_video_metadata(video_path):
    """
    Returns FPS, frame count, duration, width, height.
    """

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height
    }


# -------------------------------------------------------------
# 2. Extract frames safely
# -------------------------------------------------------------
def read_frame(cap):
    """
    Safely reads a frame and handles errors.
    Returns:
        (success_flag, frame_or_None)
    """

    if cap is None:
        return False, None

    ret, frame = cap.read()

    if not ret:
        return False, None

    return True, frame


# -------------------------------------------------------------
# 3. Load video and yield frames one by one
# -------------------------------------------------------------
def iterate_video_frames(video_path, resize=None):
    """
    Generator that yields frames one by one.
    Allows very memory-efficient processing.

    Args:
        video_path: path to input video
        resize: (width, height) if you want resized frames

    Yields:
        frame (numpy array)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize is not None:
            frame = cv2.resize(frame, resize)

        yield frame

    cap.release()


# -------------------------------------------------------------
# 4. Quick video preview utility
# -------------------------------------------------------------
def preview_video(video_path):
    """
    Displays video in a window for debugging.
    Press 'q' to quit.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    print("Previewing video... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
