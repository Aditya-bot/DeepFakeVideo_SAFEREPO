import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(signal, fs, lowcut=0.75, highcut=3.0):
    if len(signal) < fs:
        return signal

    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal)


def extract_rppg(frames, fs):
    if len(frames) < fs * 5:
        return None, None

    green_signal = []

    for face in frames:
        h, w, _ = face.shape

        # Forehead ROI
        roi = face[int(0.1*h):int(0.3*h), int(0.3*w):int(0.7*w)]

        green_channel = roi[:, :, 1]
        green_signal.append(np.mean(green_channel))

    green_signal = np.array(green_signal)

    # Normalize
    green_signal = (green_signal - np.mean(green_signal)) / (np.std(green_signal) + 1e-6)

    # Bandpass filter
    filtered = bandpass_filter(green_signal, fs)

    # FFT
    freqs = np.fft.rfftfreq(len(filtered), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(filtered))

    # Keep only valid HR range (0.75–3 Hz → 45–180 BPM)
    valid_idx = np.where((freqs >= 0.75) & (freqs <= 3.0))

    if len(valid_idx[0]) == 0:
        return None, green_signal

    peak_idx = valid_idx[0][np.argmax(fft_vals[valid_idx])]
    peak_freq = freqs[peak_idx]

    hr_bpm = peak_freq * 60

    return float(hr_bpm), filtered