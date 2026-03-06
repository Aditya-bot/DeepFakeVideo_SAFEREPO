import numpy as np

def compute_hr_confidence(signal, window=30):
    """
    Computes confidence score for HR estimate.
    
    A stable biological signal has:
    - low variance across windows
    - strong dominant peak in frequency domain

    Returns:
        confidence (0–1)
    """

    if len(signal) < window:
        return 0.0

    # Normalize signal
    sig = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    # Split into windows
    segments = []
    for i in range(0, len(sig) - window, window):
        segment = sig[i:i+window]
        segments.append(segment)

    if len(segments) == 0:
        return 0.0

    # Compute variance of each segment
    variances = [np.var(seg) for seg in segments]

    # Lower variance → more stable → more real
    stability_score = 1 / (1 + np.std(variances))

    return float(np.clip(stability_score, 0.0, 1.0))

def compute_hr_quality(hr_bpm):
    """
    Returns a quality score based on biological plausibility.
    
    - Normal HR range: 50–120 BPM
    - Deepfakes often show unrealistic peaks (0, 200+, fluctuating)
    """

    if hr_bpm is None or hr_bpm <= 0 or hr_bpm >= 200:
        return 0.0

    # Ideal HR range for a calm face on video
    if 50 <= hr_bpm <= 120:
        return 1.0

    # Reduced confidence outside normal range
    return 0.5

def estimate_hr_quality(signal, hr_bpm):
    """
    Combines confidence and biological plausibility.
    
    Returns:
        final_score (0–1)
    """

    confidence = compute_hr_confidence(signal)
    plausibility = compute_hr_quality(hr_bpm)

    final_score = 0.6 * confidence + 0.4 * plausibility

    return float(np.clip(final_score, 0.0, 1.0))
