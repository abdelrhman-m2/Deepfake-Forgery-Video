import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tempfile
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import time


# إعداد الصفحة
st.set_page_config(
    page_title="DeepFake Detective",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص CSS للتصميم
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .feature-box {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    color: #000; /* ← هذا السطر يغيّر لون الخط للأسود */
    }


    
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .real-result {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .fake-result {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 2px dashed #667eea;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# المتغيرات الثابتة
IMG_SIZE = 224
SEQ_LENGTH = 10

@st.cache_resource
def load_model():
    """تحميل النموذج المدرب"""
    try:
        # محاولة تحميل النموذج المحفوظ
        model = tf.keras.models.load_model("deepfake_detector_lstm.h5")
        return model
    except:
        # إذا لم يكن النموذج متاحاً، قم ببناء نموذج جديد
        st.warning("⚠️ النموذج المدرب غير متوفر. سيتم بناء نموذج جديد.")
        
        base_cnn = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
        base_cnn.trainable = False
        
        model = Sequential([
            TimeDistributed(base_cnn, input_shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
            TimeDistributed(GlobalAveragePooling2D()),
            LSTM(64, return_sequences=False),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def extract_frames(video_path, seq_length=SEQ_LENGTH):
    """استخراج الإطارات من الفيديو"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
    
    frame_idxs = np.linspace(0, total_frames - 1, seq_length).astype(int)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_idxs:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)
    
    cap.release()
    
    # إضافة إطارات فارغة إذا لزم الأمر
    while len(frames) < seq_length:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    
    return np.array(frames)

def predict_video(video_path, model):
    """التنبؤ بنوع الفيديو"""
    frames = extract_frames(video_path)
    
    if frames is None:
        return None, 0
    
    input_array = np.expand_dims(frames, axis=0)
    prediction = model.predict(input_array, verbose=0)
    
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    
    return class_idx, confidence

def create_confidence_chart(confidence, is_real):
    """إنشاء مخطط الثقة"""
    labels = ['Real Video', 'Fake Video']
    values = [confidence if is_real else 1-confidence, 1-confidence if is_real else confidence]
    colors = ['#4CAF50', '#f44336']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f'{v:.1%}' for v in values],
        textposition='auto',
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

def main():
    # تهيئة session state
    if 'analyzed_videos' not in st.session_state:
        st.session_state.analyzed_videos = 0
    
    # العنوان الرئيسي
    st.markdown("""
    <div class="main-header">
        <h1>🕵️ DeepFake Detective</h1>
        <p>Advanced AI-Powered Video Authenticity Analyzer</p>
        <p>Detect deepfake videos with cutting-edge machine learning technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("---")
        
        # معلومات حول النموذج
        st.markdown("### 🧠 Model Information")
        st.info("""
        **Architecture:** MobileNetV2 + LSTM
        **Input:** Video sequences (10 frames)
        **Output:** Real/Fake classification
        **Confidence:** Probability score
        """)
        
        # الإعدادات
        st.markdown("### ⚙️ Settings")
        show_frames = st.checkbox("Show extracted frames", value=True)
        show_analysis = st.checkbox("Show detailed analysis", value=True)
        
        # معلومات الاستخدام
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload** a video file (MP4, AVI, MOV)
        2. **Wait** for processing to complete
        3. **View** the results and confidence score
        4. **Analyze** the detailed breakdown
        """)
        
        # إحصائيات
        st.markdown("### 📊 Session Stats")
        st.metric("Videos Analyzed", st.session_state.analyzed_videos)
    
    # المحتوى الرئيسي
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload Video for Analysis")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file to analyze for deepfake detection"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # حفظ الملف مؤقتاً
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # عرض معلومات الفيديو
            st.markdown("### 📹 Video Information")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            with col_info1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Duration", f"{duration:.1f}s")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_info2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("FPS", f"{fps:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_info3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Frames", frame_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # عرض الفيديو
            st.markdown("### 🎬 Original Video")
            st.video(uploaded_file)
            
            # زر التحليل
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                with st.spinner("🔄 Loading AI model..."):
                    model = load_model()
                
                with st.spinner("🎯 Analyzing video for deepfake detection..."):
                    progress_bar = st.progress(0)
                    
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    class_idx, confidence = predict_video(video_path, model)
                    
                    if class_idx is not None:
                        st.session_state.analyzed_videos += 1
                        
                        is_real = class_idx == 0
                        label = "REAL" if is_real else "FAKE"
                        
                        # عرض النتيجة
                        st.markdown("### 🎯 Analysis Results")
                        
                        result_class = "real-result" if is_real else "fake-result"
                        icon = "✅" if is_real else "❌"
                        
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h2>{icon} {label}</h2>
                            <h3>Confidence: {confidence:.1%}</h3>
                            <p>{'This video appears to be authentic' if is_real else 'This video appears to be deepfake/manipulated'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # مخطط الثقة
                        if show_analysis:
                            st.markdown("### 📊 Confidence Analysis")
                            fig = create_confidence_chart(confidence, is_real)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # عرض الإطارات المستخرجة
                        if show_frames:
                            st.markdown("### 🖼️ Extracted Frames for Analysis")
                            frames = extract_frames(video_path)
                            
                            if frames is not None:
                                cols = st.columns(5)
                                for i, frame in enumerate(frames[:10]):
                                    if i < len(cols):
                                        with cols[i % 5]:
                                            # تحويل الإطار للعرض
                                            frame_display = (frame * 255).astype(np.uint8)
                                            st.image(frame_display, caption=f"Frame {i+1}", use_column_width=True)
                                    
                                    if i == 4:
                                        cols = st.columns(5)
                    else:
                        st.error("❌ Error processing video. Please try a different file.")
                
                # حذف الملف المؤقت
                os.unlink(video_path)
    
    with col2:
        st.markdown("### 🎯 Detection Features")
        
        features = [
            ("🧠", "Deep Learning", "Advanced neural network architecture"),
            ("📱", "Mobile Optimized", "Efficient MobileNetV2 backbone"),
            ("🎬", "Video Analysis", "Sequential frame processing"),
            ("⚡", "Fast Processing", "Quick and accurate detection"),
            ("🔍", "High Accuracy", "Trained on extensive datasets"),
            ("📊", "Confidence Score", "Probability-based results")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-box">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # تحذيرات مهمة
        st.markdown("### ⚠️ Important Notes")
        st.warning("""
        - This tool is for educational purposes
        - Results may vary based on video quality
        - Always verify results with multiple sources
        - Not 100% accurate - use with caution
        """)
    
    # تذييل الصفحة
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🔬 Powered by TensorFlow & Streamlit | 🚀 Built with Advanced AI Technology</p>
        <p>⚡ For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
