#this is a leagacy version of micro expression detection, it is not used in the final system but kept here for reference and future improvement ideas
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


def _landmark_distance(landmarks, idx1, idx2):
    p1 = landmarks[idx1]
    p2 = landmarks[idx2]
    return np.linalg.norm(
        np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
    )


def analyze_micro_expressions(frames):
    """
    frames: list of face frames (0–1 or uint8)
    returns score ∈ [0,1]
    """

    eye_distances = []
    eyebrow_motion = []
    lip_motion = []

    prev_eyebrow = None
    prev_lip = None

    for frame in frames:

        # FIX: handle both normalized and uint8 input
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype("uint8")
        else:
            frame_uint8 = frame.astype("uint8")

        rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        # FIX: normalize by face width
        face_width = _landmark_distance(landmarks, 234, 454) + 1e-6

        left_eye = _landmark_distance(landmarks, 160, 158) / face_width
        right_eye = _landmark_distance(landmarks, 385, 387) / face_width

        eye_distances.append((left_eye + right_eye) / 2)

        eyebrow = _landmark_distance(landmarks, 70, 105) / face_width
        lip = _landmark_distance(landmarks, 61, 291) / face_width

        # FIX: amplify motion
        if prev_eyebrow is not None:
            eyebrow_motion.append(abs(eyebrow - prev_eyebrow) * 20)

        if prev_lip is not None:
            lip_motion.append(abs(lip - prev_lip) * 20)

        prev_eyebrow = eyebrow
        prev_lip = lip

    if len(eye_distances) < 7:
        return 0.5

    # FIX: smoothing
    try:
        eye_distances = savgol_filter(eye_distances, 7, 2)
    except:
        pass

    blink_var = np.std(eye_distances)
    eyebrow_var = np.mean(eyebrow_motion) if eyebrow_motion else 0
    lip_var = np.mean(lip_motion) if lip_motion else 0

    # FIX: better scoring
    score = 0.4 * blink_var + 0.3 * eyebrow_var + 0.3 * lip_var
    score = np.clip(score * 15, 0, 1)

    return float(score)