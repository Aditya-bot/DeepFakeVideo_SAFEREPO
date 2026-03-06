import cv2
import numpy as np

# Load Haar Cascade for fast face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face(frame, target_size=(128, 128)):
    """
    Detects and extracts the largest face from a frame.

    Args:
        frame (numpy array): input BGR image
        target_size (tuple): output size (width, height)

    Returns:
        numpy array: preprocessed face image or None if not found
    """

    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)  # ensures stable detection for video
    )

    if len(faces) == 0:
        return None  # no face detected

    # Pick the largest detected face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    face = frame[y:y+h, x:x+w]

    # Resize to standard size for models
    face = cv2.resize(face, target_size)

    # Normalize to 0–1 range
    face = face.astype(np.float32) / 255.0

    return face
