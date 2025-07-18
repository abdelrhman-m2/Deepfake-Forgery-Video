# 🎭 DeepFake Video Detector 🎬

![DeepFake Detector Banner](https://img.shields.io/badge/Status-Working-green)  
🔗 Live App: [Click here](https://your-app-url.com) *(Replace this with actual ngrok or deployment URL)*

---

## 🧠 What is This Project?

This is a Deep Learning-based web app that **detects deepfake videos** using an **LSTM-based neural network** trained on frame sequences. It helps users identify manipulated videos with a **confidence score**.

---

## 🚀 Features

- 🧠 **Deep Learning**: Advanced neural network using LSTM and CNN.
- 📱 **Mobile Optimized**: Efficient MobileNetV2 as the CNN backbone.
- 🎬 **Video Analysis**: Processes sequential frames from uploaded videos.
- ⚡ **Fast Processing**: Quickly analyzes short videos (up to 8 seconds).
- 🔍 **High Accuracy**: Trained on a real/fake face dataset.
- 📊 **Confidence Score**: Shows probability-based results.

---

## 📂 Project Structure

├── app.py # Streamlit app file
├── Fake video2.ipynb # Colab notebook for model training
├── deepfake_detector_lstm.h5 # Trained deep learning model
├── requirements.txt # Python dependencies
├── Dataset.text # Dataset source/info
└── README.md # Project documentation


---

## 🔧 How to Use

### Option 1: Run in Google Colab
1. Upload all files to Colab.
2. Run all cells.
3. Click the public URL generated via `ngrok`.

### Option 2: Run Locally (Optional)
```bash
pip install -r requirements.txt
streamlit run app.py


☁️ Deploy on Hugging Face Spaces or AWS.

🙌 Credits
Made with ❤️ by Abdulrahman — for educational and research purposes.

