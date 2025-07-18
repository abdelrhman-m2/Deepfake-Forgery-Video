# 🎭 DeepFake Video Detector

🔍 A deep learning-based web application that detects deepfake videos using LSTM neural networks and the MobileNetV2 architecture.

---

## 🚀 Features

- 🧠 **Deep Learning** – Advanced LSTM and CNN architecture
- 📱 **Mobile Optimized** – Efficient MobileNetV2 backbone
- 🎬 **Video Analysis** – Frame-by-frame detection
- ⚡ **Fast Processing** – Quick and responsive inference
- 🔍 **High Accuracy** – Trained on large, curated datasets
- 📊 **Confidence Score** – Output includes probability of being fake

---

## 🛠 How It Works

1. **Upload Video** (MP4 format)
2. The video is split into frames
3. Each frame is passed through a CNN + LSTM pipeline
4. The model predicts if the video is real or fake
5. A confidence score is displayed

---
## 📁 Project Structure

├── app.py                  # Streamlit frontend
├── deepfake_detector_lstm.h5   # Pretrained model
├── requirements.txt        # Python dependencies
├── Dataset.text            # Video dataset links/info
├── Fake video2.ipynb       # Training & preprocessing notebook
└── README.md               # This file

## 📦 Installation

```bash
git clone https://github.com/yourusername/deepfake-detector
cd deepfake-detector
pip install -r requirements.txt
