
import streamlit as st
import torch
import librosa
import numpy as np
import joblib
import torch.nn as nn
import tempfile
import os

# ✅ Page Configuration
st.set_page_config(
    page_title="VoiceScope India | Accent Classifier",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# 🎨 ULTRA-MODERN CSS STYLING
# ===============================
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }

.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    animation: bgflow 15s ease infinite;
    background-size: 400% 400%;
}
@keyframes bgflow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* 🎤 Hero Header Animation */
.hero-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    padding: 5rem 2rem;
    border-radius: 30px;
    text-align: center;
    color: white;
    margin-bottom: 3.5rem;
    box-shadow: 0 25px 70px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: rotate 25s linear infinite;
}
@keyframes rotate { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

.hero-header h1 {
    font-size: 4rem;
    margin: 0;
    font-weight: 700;
    text-shadow: 0 4px 10px rgba(0,0,0,0.4);
    background: linear-gradient(45deg, #fff, #ffd700, #fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s ease-in-out infinite;
}
@keyframes shine {
    0%,100% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
}
.hero-subtitle {
    font-size: 1.4rem;
    margin-top: 1rem;
    opacity: 0.9;
    animation: fadeIn 2s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* 🎯 Glassmorphism Upload Card */
.glass-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2.5rem;
    margin: 2rem 0;
    border: 1px solid rgba(255,255,255,0.3);
    text-align: center;
    color: black;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    transition: all 0.4s ease;
}
.glass-card:hover { transform: translateY(-10px) scale(1.02); }

/* 🎵 Floating Icons */
.float-icon {
    font-size: 4rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-15px); }
}

/* 🎚️ Result Box Animation */
.result-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 4rem 2rem;
    border-radius: 30px;
    margin: 2.5rem 0;
    color: white;
    text-align: center;
    box-shadow: 0 25px 70px rgba(0,0,0,0.4);
    animation: slideUp 0.8s ease-out;
}
@keyframes slideUp {
    from {opacity: 0; transform: translateY(60px);}
    to {opacity: 1; transform: translateY(0);}
}
.result-box h2 {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    animation: pulse 2.5s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% {transform: scale(1);}
    50% {transform: scale(1.05);}
}

/* 🔥 Confidence Bar */
.confidence-bar {
    background: rgba(255,255,255,0.25);
    height: 22px;
    border-radius: 12px;
    margin: 1.5rem auto;
    max_width: 500px;
    overflow: hidden;
    box_shadow: inset 0 0 10px rgba(255,255,255,0.4);
}
.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #fff, #ffd700);
    animation: fillAnim 2s ease-out forwards;
}
@keyframes fillAnim {
    from { width: 0%; }
}

/* 🍛 Cuisine Cards */
.cuisine-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    box-shadow: 0 15px 40px rgba(102,126,234,0.5);
    text-align: center;
    transition: all 0.4s ease;
}
.cuisine-card:hover {
    transform: translateY(-10px) rotate(1deg);
    box-shadow: 0 20px 60px rgba(118,75,162,0.7);
}

/* 👨‍💻 Team Section */
.team-credits {
    background: linear-gradient(135deg, #000000 0%, #434343 100%);
    padding: 4rem 2rem;
    border-radius: 30px;
    color: white;
    text-align: center;
    margin-top: 4rem;
    box-shadow: 0 25px 70px rgba(0,0,0,0.5);
}
.team-member {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    padding: 1rem 2rem;
    margin: 0.5rem;
    border-radius: 25px;
    border: 2px solid rgba(255,255,255,0.3);
    transition: all 0.4s ease;
}
.team-member:hover {
    transform: scale(1.08);
    border-color: #ffd700;
}

/* Hide branding */
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🎤 HERO HEADER
# ===============================
st.markdown(r"""
<div class="hero-header">
    <h1>🎤 VoiceScope India</h1>
    <p class="hero-subtitle">AI-Powered Regional Accent Classifier & Cultural Discovery</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# 🧠 MODEL DEFINITION
# ===============================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

@st.cache_resource
def load_pipeline():
    scaler = joblib.load('scaler.pkl')
    # max_mfcc_len is now saved within the scaler object
    max_mfcc_len = scaler.max_mfcc_len
    input_dim = scaler.n_features_in_
    model = SimpleMLP(input_dim, 128, 6)
    model.load_state_dict(torch.load('accent_model.pt', map_location='cpu'))
    model.eval()
    return model, scaler, max_mfcc_len

# Use the max_mfcc_len value obtained from the previous execution
max_mfcc_len_global = 701 # This value comes from the previous cell's output

@st.cache_resource
def load_pipeline_updated(max_mfcc_len_val):
    scaler = joblib.load('scaler.pkl')
    # max_mfcc_len is now saved within the scaler object
    max_mfcc_len = max_mfcc_len_val # Use the global variable
    input_dim = scaler.n_features_in_
    model = SimpleMLP(input_dim, 128, 6)
    model.load_state_dict(torch.load('accent_model.pt', map_location='cpu'))
    model.eval()
    return model, scaler, max_mfcc_len

model, scaler, max_mfcc_len = load_pipeline_updated(max_mfcc_len_global)


# ===============================
# 🗺️ REGION MAPPING
# ===============================
region_to_cuisine = {
    0: ("North India 🏔️", ["Butter Chicken & Naan", "Dal Makhani", "Chole Bhature"], "🏔️"),
    1: ("South India 🌴", ["Idli-Sambar", "Masala Dosa", "Fish Curry"], "🌴"),
    2: ("East India 🌊", ["Hilsa Curry", "Litti-Chokha", "Mishti Doi"], "🌊"),
    3: ("West India 🏜️", ["Dhokla", "Vindaloo", "Vada Pav"], "🏜️"),
    4: ("Northeast India ⛰️", ["Smoked Pork", "Fish Tenga", "Momo"], "⛰️"),
    5: ("Central India 🌾", ["Poha", "Bhutte Ki Kees", "Bafauri"], "🌾")
}

# ===============================
# 🎧 FEATURE EXTRACTION + PREDICTION
# ===============================
def extract_mfcc_features(wav, sr, n_mfcc=40, hop_length=512, n_fft=2048, max_len=None):
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    if max_len:
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
    return mfcc.flatten()

def predict_from_audio(audio_data, sr):
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=0)
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    features = extract_mfcc_features(audio_data, 16000, max_len=max_mfcc_len)
    features_scaled = scaler.transform(features.reshape(1, -1))
    with torch.no_grad():
        outputs = model(torch.FloatTensor(features_scaled))
        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0][pred_idx].item()
    return pred_idx, confidence


# ===============================
# 📁 UPLOAD SECTION
# ===============================
st.markdown(r"""
<div class="glass-card">
    <div class="float-icon">🎧</div>
    <h3>📁 Upload Your Audio</h3>
    <p>Supported formats: WAV, MP3, M4A — min 1 sec of clear speech</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")

if uploaded_file:
    st.audio(uploaded_file)
    try:
        with st.spinner("🔍 Analyzing your accent with Deep Learning..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            audio_data, sr = librosa.load(tmp_path, sr=None)
            os.unlink(tmp_path)
            if len(audio_data) < sr * 0.5:
                st.error("❌ Audio too short! Please upload at least 1 second.")
            else:
                pred_idx, confidence = predict_from_audio(audio_data, sr)
                region, cuisines, icon = region_to_cuisine[pred_idx]

                # 🎯 RESULT SECTION
                st.markdown(f"""
                <div class='result-box'>
                    <div style='font-size:5rem;'>{icon}</div>
                    <h2>{region}</h2>
                    <p style='font-size:1.3rem;'>Detected Accent Region</p>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width:{confidence*100}%;'></div>
                    </div>
                    <p style='font-size:1.2rem;'>Confidence: {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # 🍽️ CUISINE SECTION
                st.markdown(r"""
                <div class="glass-card">
                    <h3>🍛 Regional Cuisine Highlights</h3>
                    <p>Traditional dishes from your detected region:</p>
                </div>
                """, unsafe_allow_html=True)

                cols = st.columns(len(cuisines))
                for col, dish in zip(cols, cuisines):
                    col.markdown(f"""
                    <div class='cuisine-card'>
                        <strong>🌟 {dish}</strong>
                        <p>{region.split()[0]} Specialty Dish</p>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ===============================
# 👨‍💻 TEAM SECTION
# ===============================
st.markdown(r"""
<div class="team-credits">
    <h3>👨‍💻 Developed By</h3>
    <div>
        <div class="team-member">🎓 Pangoth Hemanth Nayak</div>
        <div class="team-member">🎓 Arutla Prasanna</div>
        <div class="team-member">🎓 Apurba Nandi</div>
    </div>
    <p style="margin-top:2rem;opacity:0.8;">🏛️ IIIT Hyderabad | NLP Final Project 2025</p>
    <p style="opacity:0.7;">Powered by PyTorch • Librosa • Streamlit • Neural Networks</p>
</div>
""", unsafe_allow_html=True)
