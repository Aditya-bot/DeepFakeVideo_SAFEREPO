import numpy as np
from scipy.signal import find_peaks


def estimate_hr_quality(signal, hr_bpm, fs):
    """
    Robust HR quality estimation
    """

    if signal is None or len(signal) < fs * 3:
        return 0.0

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    # ---- PEAK DETECTION ----
    peaks, _ = find_peaks(signal, distance=fs/2)

    if len(peaks) < 3:
        return 0.0

    # ---- INTERVAL CONSISTENCY ----
    intervals = np.diff(peaks) / fs
    interval_std = np.std(intervals)

    consistency_score = np.exp(-interval_std * 3)

    # ---- HR PLAUSIBILITY ----
    if hr_bpm is None or hr_bpm < 40 or hr_bpm > 180:
        hr_score = 0.0
    elif 50 <= hr_bpm <= 120:
        hr_score = 1.0
    else:
        hr_score = 0.5

    # ---- SIGNAL ENERGY CHECK ----
    energy = np.std(signal)
    energy_score = np.clip(energy, 0, 1)

    # ---- FINAL SCORE ----
    final_score = (
        0.5 * consistency_score +
        0.3 * hr_score +
        0.2 * energy_score
    )

    return float(np.clip(final_score, 0.0, 1.0))