import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# -------------------------------------------------------------
# 1. SIGNAL NORMALIZATION
# -------------------------------------------------------------
def normalize_signal(signal):
    """
    Normalizes a signal to zero mean and unit variance.
    """
    signal = np.array(signal)

    if len(signal) == 0:
        return signal

    mean = np.mean(signal)
    std = np.std(signal) + 1e-6   # avoid division by zero

    return (signal - mean) / std


# -------------------------------------------------------------
# 2. SMOOTHING FILTERS
# -------------------------------------------------------------
def moving_average(signal, window=5):
    """
    Applies moving average smoothing.
    """
    if len(signal) < window:
        return np.array(signal)

    return np.convolve(signal, np.ones(window) / window, mode='same')


# -------------------------------------------------------------
# 3. BANDPASS FILTER (for rPPG)
# -------------------------------------------------------------
def butter_bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30, order=5):
    """
    Applies Butterworth bandpass filter to isolate pulse frequencies.
    Heart rate typically lies in 0.7–4.0 Hz (42–240 BPM).
    """

    if len(signal) < fs:
        # Too short to filter meaningfully
        return np.array(signal)

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    return filtered


# -------------------------------------------------------------
# 4. FFT-BASED HEART RATE ESTIMATION
# -------------------------------------------------------------
def estimate_hr_fft(signal, fs=30):
    """
    Estimates heart rate from rPPG signal using FFT.

    Args:
        signal (array-like): filtered rPPG signal
        fs (int): sampling rate = frames per second

    Returns:
        hr_bpm (float): estimated heart rate in beats per minute
        freqs (np.array): frequency bins
        fft_vals (np.array): magnitude spectrum
    """

    signal = np.array(signal)

    if len(signal) < fs:
        return None, None, None

    # FFT
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))

    # Remove very low frequency noise
    fft_vals[0:1] = 0

    # Peak frequency
    peak_idx = np.argmax(fft_vals)
    peak_freq = freqs[peak_idx]

    hr_bpm = peak_freq * 60.0  # convert Hz → BPM

    if hr_bpm <= 0 or hr_bpm >= 200:
        # Out of human range → treat as invalid
        return None, freqs, fft_vals

    return hr_bpm, freqs, fft_vals


# -------------------------------------------------------------
# 5. PEAK DETECTION (optional use)
# -------------------------------------------------------------
def detect_peaks(signal, distance=5):
    """
    Detects peaks in the signal.
    Useful for HR estimation using time-domain methods.
    """
    signal = np.array(signal)
    peaks, _ = find_peaks(signal, distance=distance)
    return peaks
