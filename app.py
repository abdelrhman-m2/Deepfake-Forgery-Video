import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
import tempfile
import os
from PIL import Image
import plotly.graph_objects as go
import time

# ✅ تعطيل تحذيرات TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# إعداد الصفحة
st.set_page_config(
    page_title="DeepFake Detective",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ تحميل النموذج
@st.cache_resource
def load_model():
    try:
        model = keras_load_model("deepfake_detector_lstm.h5")
        return model
    except Exception as e:
        st.error("⚠️ النموذج غير متوفر أو غير صالح. يرجى رفع ملف النموذج المدرب.")
        return None

# ✅ الدالة لاستخراج الإطارات
def extract_frames(video_path, seq_length=10, img_size=224):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, seq_length).astype(int)

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    while len(frames) < seq_length:
        frames.append(np.zeros((img_size, img_size, 3)))

    return np.array(frames)

# ✅ التنبؤ بالفيديو
def predict_video(video_path, model):
    frames = extract_frames(video_path)
    if frames is None or model is None:
        return None, 0
    input_array = np.expand_dims(frames, axis=0)
    prediction = model.predict(input_array, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    return class_idx, confidence

# ✅ رسم نتائج الثقة
def create_confidence_chart(confidence, is_real):
    labels = ['Real Video', 'Fake Video']
    values = [confidence if is_real else 1 - confidence,
              1 - confidence if is_real else confidence]
    colors = ['#4CAF50', '#f44336']
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f'{v:.1%}' for v in values],
        textposition='auto'
    )])
    fig.update_layout(
        title="Confidence Analysis",
        yaxis_title="Confidence Level",
        xaxis_title="Prediction",
        font=dict(size=14),
        height=400,
        showlegend=False
    )
    return fig

# ✅ الواجهة الرئيسية
def main():
    st.title("🕵️ DeepFake Video Detector")
    st.write("Upload a video file to check if it's **REAL** or **FAKE** using a trained LSTM-based model.")

    uploaded_file = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(uploaded_file)

        if st.button("🔍 Analyze"):
            with st.spinner("Loading model and analyzing..."):
                model = load_model()
                if model is None:
                    return
                class_idx, confidence = predict_video(video_path, model)
                if class_idx is not None:
                    label = "REAL ✅" if class_idx == 0 else "FAKE ❌"
                    is_real = class_idx == 0
                    st.success(f"Prediction: **{label}** with confidence **{confidence:.1%}**")
                    fig = create_confidence_chart(confidence, is_real)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to process the video.")

        os.unlink(video_path)

if __name__ == "__main__":
    main()
