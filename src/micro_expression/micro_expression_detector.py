import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


def _landmark_distance(landmarks, idx1, idx2):
    p1 = landmarks[idx1]
    p2 = landmarks[idx2]
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))


def analyze_micro_expressions(frames):
    """
    frames: list of face frames
    returns score ∈ [0,1]
    """

    eye_distances = []
    eyebrow_motion = []
    lip_motion = []

    prev_eyebrow = None
    prev_lip = None

    for frame in frames:

        frame_uint8 = (frame * 255).astype("uint8")
        rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = _landmark_distance(landmarks, 160, 158)
        right_eye = _landmark_distance(landmarks, 385, 387)

        eye_distances.append((left_eye + right_eye) / 2)

        eyebrow = _landmark_distance(landmarks, 70, 105)

        if prev_eyebrow is not None:
            eyebrow_motion.append(abs(eyebrow - prev_eyebrow))

        prev_eyebrow = eyebrow

        lip = _landmark_distance(landmarks, 61, 291)

        if prev_lip is not None:
            lip_motion.append(abs(lip - prev_lip))

        prev_lip = lip

    if len(eye_distances) < 5:
        return 0.5

    blink_var = np.std(eye_distances)
    eyebrow_var = np.mean(eyebrow_motion) if eyebrow_motion else 0
    lip_var = np.mean(lip_motion) if lip_motion else 0

    motion_score = blink_var + eyebrow_var + lip_var

    score = np.tanh(motion_score * 10)

    return float(score)