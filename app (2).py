# app.py → CLEAN & FINAL (only model, no extra text)
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="World Risk Index Calculator",
    page_icon="globe",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    path = "WRI_WINNER_MODEL_2025.pkl"
    if not os.path.exists(path):
        st.error("Model file missing: WRI_WINNER_MODEL_2025.pkl")
        st.stop()
    return joblib.load(path)

data = load_model()
base_models = data["base_models"]
meta_model = data["meta_model"]

st.title("World Risk Index Calculator")
st.markdown("---")

c1, c2, c3 = st.columns(3)
with c1:
    exposure = st.slider("Exposure", 1, 100, 50)
    vulnerability = st.slider("Vulnerability", 1, 100, 60)
with c2:
    susceptibility = st.slider("Susceptibility", 1, 100, 55)
    lack_coping = st.slider("Lack of Coping Capabilities", 1, 100, 70)
with c3:
    lack_adaptive = st.slider("Lack of Adaptive Capacities", 1, 100, 65)

user_input = np.array([[exposure, vulnerability, susceptibility, lack_coping, lack_adaptive]])

level1 = np.column_stack([m.predict(user_input)[0] for m in base_models])
wri = round(float(meta_model.predict(level1)[0]), 4)

st.markdown("---")
st.metric("World Risk Index", wri)

if wri >= 25:
    st.error("Extremely High Risk")
elif wri >= 18:
    st.warning("Very High Risk")
elif wri >= 12:
    st.warning("High Risk")
elif wri >= 7:
    st.info("Medium Risk")
else:
    st.success("Low / Very Low Risk")

# Optional country match
matches = {"Vanuatu":32.0, "Philippines":24.3, "Tonga":29.1, "Japan":6.2, "Netherlands":1.5}
for country, score in matches.items():
    if abs(wri - score) <= 3:
        st.write(f"→ Similar to **{country}** ({score})")