import streamlit as st
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import requests

# ================== LOAD ANIMATION ==================
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

emotion_anim = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_w51pcehl.json")  # Futuristic animation
emoji_map = {
    'sadness': "😢",
    'anger': "😠",
    'love': "❤️",
    'surprise': "😲",
    'fear': "😨",
    'joy': "😊"
}

# ================== LOAD MODEL & VECTOR ==================
model = joblib.load("emotion_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Label mapping
label_map = {
    0: 'sadness',
    1: 'anger',
    2: 'love',
    3: 'surprise',
    4: 'fear',
    5: 'joy'
}

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="Emotion Detector", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    .big-title {
        font-size:40px !important;
        font-weight:bold;
        text-align:center;
        color: #00f5d4;
        text-shadow: 1px 1px 5px #222;
    }
    .predicted {
        font-size:25px !important;
        font-weight:bold;
        color: #fff;
        background: linear-gradient(to right, #00f5d4, #0aff99);
        padding: 8px 15px;
        border-radius: 10px;
        display:inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== HEADER ==================
st.markdown('<p class="big-title">🚀 Futuristic Emotion Detection App</p>', unsafe_allow_html=True)
st.write("Type your text below and watch AI predict your **emotion** in style!")

# ================== ANIMATION ==================
if emotion_anim:
    st_lottie(emotion_anim, height=200, key="emotion_anim")

# ================== USER INPUT ==================
user_input = st.text_area("💬 Enter your text:")

if st.button("🔮 Predict Emotion"):
    if user_input.strip():
        # Transform and predict
        text_vector = tfidf.transform([user_input])
        prediction = model.predict(text_vector)[0]
        proba = model.predict_proba(text_vector)[0]

        emotion = label_map[prediction]
        confidence = np.max(proba) * 100

        st.markdown(f"<p class='predicted'>{emoji_map[emotion]} Predicted Emotion: {emotion} ({confidence:.2f}%)</p>", unsafe_allow_html=True)

        # Show probability bar chart
        st.subheader("📊 Confidence Levels")
        st.bar_chart({label_map[i]: [proba[i]*100] for i in range(len(proba))})
    else:
        st.warning("⚠ Please enter some text.")

# ================== FOOTER ==================
st.markdown("---")
st.markdown("✨ Built with Streamlit, TF-IDF & Logistic Regression | UI by **ALI SHAN** 🚀")
