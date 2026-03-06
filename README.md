
# 🧬 Deepfake Detection Using rPPG & Deep Learning

A robust hybrid system that detects deepfake videos by analyzing **physiological signals (remote Photoplethysmography – rPPG)** along with **deep learning–based visual fake detection models**.  
This project merges **biological cues** with **frame-level CNN/Transformer predictions** to significantly improve deepfake detection reliability.

---

## Features

### 🔹 Physiological Signal Extraction
- Extract rPPG signals from green-channel variations.
- Estimate heart rate & signal stability.
- Identify physiological inconsistencies typical in deepfakes.

### 🔹 Deepfake Detection Models
- CNN-based frame-level classifier.
- Optional Transformer-based temporal analysis.
- Detects texture artifacts, blending issues, and motion inconsistencies.

### 🔹 Fusion Engine
- Combines visual predictions + physiological metrics.
- Increases robustness on unseen deepfake techniques.

### 🔹 End-to-End Automated Pipeline
- Input video → Face Tracking → rPPG Extraction → CNN Prediction → Final Decision.

---

## 🏗 Project Structure

```

Deepfake_Detection_Project/
│
├── data/
│   ├── real/
│   ├── fake/
│   └── samples/
│
├── models/
│   ├── rppg/
│   ├── deepfake_cnn/
│   ├── face_recognition/
│
├── src/
│   ├── preprocessing/
│   │     ├── face_extraction.py
│   │     ├── frame_generator.py
│   │
│   ├── rppg/
│   │     ├── rppg_extractor.py
│   │     ├── heart_rate_estimator.py
│   │
│   ├── deepfake/
│   │     ├── cnn_detector.py
│   │     ├── transformer_detector.py
│   │
│   ├── fusion/
│   │     └── decision_fusion.py
│   │
│   ├── utils/
│         ├── video_utils.py
│         ├── signal_utils.py
│
├── main.py
├── requirements.txt
└── README.md

```

---

## Installation & Setup

### 1️⃣ Create virtual environment
```

python -m venv venv

```

### 2️⃣ Activate environment
#### Windows:
```

venv\Scripts\activate

```
#### Mac/Linux:
```

source venv/bin/activate

```

### 3️⃣ Install dependencies
```

pip install -r requirements.txt

```

---

## ▶️ Running the Project

To execute the full pipeline:

```

python main.py

```

This performs:
1. Face extraction  
2. rPPG heart rate estimation  
3. Deepfake probability prediction  
4. Final decision fusion  

---

## 🔧 How the System Works

### **1. Face Detection & Preprocessing**
- Extracts face ROI using Haar Cascades or Mediapipe.
- Normalizes frames for rPPG and CNN input.

### **2. rPPG Signal Extraction**
- Computes the average green-channel intensity per frame.
- Converts the temporal waveform into an estimated heart rate.
- Deepfakes show unstable biological rhythms.

### **3. Deepfake Model Prediction**
- CNN analyzes texture, blending, edge inconsistencies.
- Optional Transformer model analyzes temporal coherence.

### **4. Fusion Layer**
Combines:
- Deepfake probability  
- Heart rate stability  
- Signal quality index  

Outputs **Real / Fake** classification.

---

## Recommended Datasets

### Deepfake datasets:
- FaceForensics++
- Celeb-DF
- DFDC
- DeepFake-TIMIT
- FakeAVCeleb

### Physiological datasets (optional):
- PURE
- VIPL-HR

---

## Future Enhancements
- Add micro-expression detection  
- Build Streamlit GUI  
- Deploy as cloud API  
- Add LSTM-based temporal fusion  
- Optimize for real-time inference  

---

## Contributing
Pull requests and suggestions are welcome!

