# 🧬 DeepScan — Multimodal Deepfake Video Detection

A robust hybrid system that detects deepfake videos by integrating **spatial**, **motion-based**, and **physiological** features through a unified multimodal architecture.  
DeepScan fuses a **ResNet-18 CNN**, a **Micro-CNN on inter-frame difference maps**, and a **Remote Photoplethysmography (rPPG)** module via a weighted decision fusion formula to deliver reliable deepfake classification.

> **Conference Submission:** Multimodal Deepfake Video Detection Using Spatial, Motion-Based, and Physiological Feature Integration — *AISIGHSD2026 (Paper ID: 170)*

---

## Features

### 🔹 Spatial Feature Extraction (CNN)
- ResNet-18-based frame-level classifier.
- Detects texture artifacts, blending inconsistencies, and boundary-level manipulations.
- Contributes **50%** weight in the final fusion decision.

### 🔹 Motion-Based Micro-Expression Analysis
- Computes amplified inter-frame difference maps.
- Micro-CNN detects subtle motion inconsistencies characteristic of synthetic facial animations.
- Contributes **30%** weight in the final fusion decision.

### 🔹 Physiological Signal Extraction (rPPG)
- Extracts rPPG signals from green-channel intensity variations across frames.
- Estimates heart rate and signal stability index.
- Identifies physiological inconsistencies typically present in deepfake videos.
- Contributes **20%** weight in the final fusion decision.

### 🔹 Weighted Decision Fusion Engine
- Combines all three modalities using the formula:

  **Score = 0.5 × CNN + 0.3 × Micro-CNN + 0.2 × rPPG**

- Classification threshold: **0.6** (≥ 0.6 → Fake, < 0.6 → Real)
- Increases robustness against unseen deepfake techniques.

### 🔹 Streamlit Web Interface
- Cyber/forensic-lab themed UI with custom CSS and scanline overlay.
- End-to-end pipeline: upload video → face tracking → feature extraction → classification → result display.

---

## 📊 Experimental Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~83%   |
| F1-Score  | ~79%   |
| Test Samples | 179 videos |

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
│   │     └── transformer_detector.py
│   │
│   ├── micro_expression/
│   │     └── micro_expression_detector.py
│   │
│   ├── fusion/
│   │     └── decision_fusion.py
│   │
│   └── utils/
│         ├── video_utils.py
│         └── signal_utils.py
│
├── app.py                  # Streamlit UI
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

### Full pipeline (CLI):
```
python main.py
```

This performs:
1. Face extraction
2. rPPG heart rate estimation
3. Micro-expression motion analysis
4. Deepfake probability prediction via CNN
5. Weighted decision fusion → Final classification

### Streamlit UI:
```
streamlit run app.py
```

---

## 🔧 How the System Works

### **1. Face Detection & Preprocessing**
- Extracts face ROI using Haar Cascades or MediaPipe.
- Normalizes frames for rPPG, Micro-CNN, and CNN input.

### **2. Spatial Analysis (ResNet-18 CNN)**
- Analyzes texture artifacts, blending boundaries, and compression noise.
- Outputs a deepfake probability score.

### **3. Motion-Based Micro-Expression Analysis**
- Computes amplified inter-frame difference maps.
- Micro-CNN processes these maps to detect unnatural micro-movements.
- Outputs a motion anomaly score.

### **4. Physiological Signal Extraction (rPPG)**
- Computes average green-channel intensity per frame to extract a temporal waveform.
- Estimates heart rate and signal quality index.
- Deepfakes often exhibit unstable or absent biological rhythms.

### **5. Weighted Decision Fusion**
Combines all three scores:

```
Final Score = 0.5 × CNN_score + 0.3 × Micro_score + 0.2 × rPPG_score
Decision    = "Fake" if Final Score ≥ 0.6 else "Real"
```

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
- Add LSTM-based temporal fusion for longer video sequences
- Improve rPPG robustness under variable lighting and compression
- Deploy as a cloud API for real-time inference
- Expand training dataset size for micro-expression and physiological branches
- Explore attention-based fusion mechanisms

---

## Authors
- **Aditya Suresh** — REVA University
- **Darshan A Jain** — REVA University
- **Nihith A Naik** — REVA University
- **Himanshu V** — REVA University

*Supervised by Dr. Manisha Swasthik, School of CSE, REVA University*

---

## Contributing
Pull requests and suggestions are welcome!