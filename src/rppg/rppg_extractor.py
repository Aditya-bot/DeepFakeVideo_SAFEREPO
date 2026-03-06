import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30):
    """
    Applies a bandpass filter to isolate heart-rate frequencies.
    HR range 0.7–4 Hz = 42–240 BPM.
    """
    if len(signal) < fs:  
        return signal  # too short to filter

    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal)

def extract_rppg(frames):
    """
    Extracts rPPG signal using the Green Channel Method.
    
    Args:
        frames: list of face frames (normalized, 0–1)

    Returns:
        (float, np.array): estimated HR (BPM), raw signal array
    """

    if len(frames) == 0:
        return None, None

    # Step 1: Extract average green intensity per frame
    green_signal = []
    for face in frames:
        green_channel = face[:, :, 1]  # index 1 = green channel
        green_signal.append(np.mean(green_channel))

    green_signal = np.array(green_signal)

    # Step 2: Apply bandpass filtering to isolate pulse wave
    fs = 30  # assume 30 FPS input video
    filtered = bandpass_filter(green_signal, fs=fs)

    # Step 3: Estimate heart rate using FFT
    freqs = np.fft.rfftfreq(len(filtered), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(filtered))

    # Ignore very low frequencies
    fft_vals[0:1] = 0

    # Find dominant frequency
    peak_idx = np.argmax(fft_vals)
    peak_freq = freqs[peak_idx]  # Hz

    heart_rate_bpm = peak_freq * 60  # convert to BPM

    return heart_rate_bpm, green_signal
