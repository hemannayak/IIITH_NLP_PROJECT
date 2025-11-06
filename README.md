# IIITH_NLP_PROJECT


# 🎙️ **VoiceScope India: Regional Accent Classifier & Cultural Discovery**

> *“Every accent tells a story — and we’re here to listen.”*
> A deep learning project celebrating India’s **diverse voices and cultures**, turning speech into stories and flavors 🍛

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Neural%20Network-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" />
  <img src="https://img.shields.io/github/stars/<your-username>/VoiceScope-India?style=social" />
</p>

---

## 🌏 **Overview**

**VoiceScope India** is a **regional accent classification system** powered by AI.
You upload a short voice clip, and it:
🎧 Identifies your **regional accent**
🍱 Suggests **a traditional dish** from that region
🧠 Helps celebrate **India’s linguistic and cultural diversity**

Built with 💻 **Python**, 🧠 **PyTorch**, and 🌐 **Streamlit**, this project blends ML precision with cultural creativity.

---

## 📸 **Project Preview**

<p align="center">
  <img src="https://github.com/<your-username>/VoiceScope-India/blob/main/assets/voicescope_preview.gif" alt="App Demo" width="700"/>
</p>

> *A glimpse of our Streamlit app predicting accents & showing regional dishes.*

---

## 🎧 **Dataset: IndicAccentDb**

| Feature              | Description                                                                        |
| -------------------- | ---------------------------------------------------------------------------------- |
| 📦 **Source**        | [DarshanaS/IndicAccentDb](https://huggingface.co/datasets/DarshanaS/IndicAccentDb) |
| 🎙️ **Audio Format** | `.wav` at 16kHz sampling rate                                                      |
| 🗣️ **Accents**      | Hindi, Bengali, Tamil, Telugu, Kannada, Gujarati, Punjabi, and more                |
| 🔢 **Samples**       | Thousands of labeled audio clips                                                   |
| 🧹 **Preprocessing** | Resampling → Mono conversion → Noise cleaning → Normalization → Padding/Truncation |

---

## 🧠 **Methodology**

### 🎵 **Feature Extraction (MFCCs)**

We extracted **Mel-Frequency Cepstral Coefficients (MFCCs)** — features that capture human hearing perception.

```python
n_mfcc = 40
hop_length = 512
n_fft = 2048
max_mfcc_len = 200
```

Shorter audios were **padded**, longer ones **trimmed** for uniform input.

---

### 🧩 **Model Architecture: SimpleMLP**

| Layer | Type                     | Activation |
| ----- | ------------------------ | ---------- |
| 1     | Linear (Input → Hidden)  | ReLU       |
| 2     | Linear (Hidden → Hidden) | ReLU       |
| 3     | Dropout (0.3)            | -          |
| 4     | Linear (Hidden → Output) | Softmax    |

💡 Trained using `Adam` optimizer and `CrossEntropyLoss` for multi-class classification.

---

### 🏋️ **Training Setup**

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Optimizer     | Adam                       |
| Learning Rate | 0.001                      |
| Epochs        | 50                         |
| Batch Size    | 32                         |
| Split         | 80% train / 20% validation |
| Framework     | PyTorch                    |

---

### 📈 **Evaluation Metrics**

✅ Accuracy
🎯 Precision
🔁 Recall
🧮 F1-Score

---

## 📊 **Model Performance**

```
              precision    recall  f1-score   support
Bengali         0.87       0.84       0.85        60
Tamil           0.82       0.79       0.81        55
Hindi           0.89       0.91       0.90        70
Overall Accuracy: 0.86
```

**Confusion Matrix Insight:** Tamil and Telugu accents occasionally overlap due to phonetic similarities — a real-world linguistic challenge!

---

## 🧰 **Tech Stack**

| Category             | Tools               |
| -------------------- | ------------------- |
| 💻 Language          | Python              |
| 🧠 ML Framework      | PyTorch             |
| 🎚️ Audio Processing | Librosa, Soundfile  |
| 🗂️ Data Handling    | NumPy, Pandas       |
| 🧮 Persistence       | Joblib              |
| 🌐 Web Framework     | Streamlit           |
| ☁️ Deployment        | Streamlit Cloud     |
| 📊 Visualization     | Matplotlib, Seaborn |

---

## 🚀 **Streamlit App Experience**

1. 🎤 Upload your voice (wav/mp3/ogg)
2. 🧠 Model preprocesses, extracts MFCCs, scales features
3. ⚡ Predicts the accent using `accent_model.pt`
4. 🍲 Displays a traditional **dish suggestion** from that region

Example:

> Input: *Telugu accent* → Output: *Region: Andhra Pradesh* → *Dish: Gongura Pachadi 🌿*

---

## 🗂️ **Repository Structure**

```
📁 VoiceScope-India/
├── app.py                 # Streamlit main app
├── accent_model.pt        # Trained PyTorch model
├── scaler.pkl             # Feature scaler
├── requirements.txt       # Dependencies
├── AccentClassifier.ipynb # Model training notebook
├── README.md              # This file 😎
└── assets/
    └── voicescope_preview.gif
```

---

## 💻 **Run Locally**

```bash
# Clone the repo
git clone https://github.com/<your-username>/VoiceScope-India.git

# Enter directory
cd VoiceScope-India

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Then open: 👉 [http://localhost:8501](http://localhost:8501)

---

## ☁️ **Deployment**

Hosted on **Streamlit Community Cloud** ☁️

> Zero setup. One click. Full cultural exploration.
> 🔗 [Live App](https://voicescopeindia.streamlit.app) *(replace once deployed)*

---

## 👨‍💻 **Developed By**

| Name                         | Roll No    | College |
| ---------------------------- | ---------- | ------- |
| 🎓 **Pangoth Hemanth Nayak** | 23E51A67C5 | HITAM   |
| 🎓 **Arutla Prasanna**       | 23E51A6711 | HITAM   |
| 🎓 **Apurba Nandi**          | 23E51A6708 | HITAM   |

---

## 💖 **Acknowledgements**

Gratitude to:

* 🗃️ [DarshanaS/IndicAccentDb](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)
* 🧑‍🏫 Mentors who inspired the vision
* ☕ Late-night caffeine and the open-source community 💻

---

> ✨ *“India speaks in many tongues. VoiceScope just helps you listen closer.”*

