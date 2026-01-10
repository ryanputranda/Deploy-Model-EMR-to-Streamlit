import streamlit as st
import joblib
import numpy as np

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="EMR Prediction",
    layout="centered"
)

# ======================
# Custom CSS
# ======================
st.markdown("""
<style>
body {
    background-color: #f3e8ff;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f3e8ff;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
}
.footer img {
    height: 20px;
    vertical-align: middle;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# Load model & encoders
# ======================
model = joblib.load("random_forest_model.pkl")

le_diagnosa   = joblib.load("le_diagnosa.pkl")
le_institusi  = joblib.load("le_institusi.pkl")
le_area       = joblib.load("le_area.pkl")
le_tipe       = joblib.load("le_tipe.pkl")
le_biaya      = joblib.load("le_biaya.pkl")

# ======================
# Title
# ======================
st.title(
    "Segmentasi Pegawai berdasarkan Data EMR (Rekam Medis) Menggunakan GridSearchCV - Random Forest Classifier"
)

# ======================
# Input (langsung dari encoder)
# ======================
diagnosa_input = st.selectbox(
    "Choose Diagnose",
    le_diagnosa.classes_
)

institusi_input = st.selectbox(
    "Choose Institution",
    le_institusi.classes_
)

area_input = st.selectbox(
    "Choose Area",
    le_area.classes_
)

tipe_input = st.selectbox(
    "Choose Type",
    le_tipe.classes_
)

biaya_input = st.selectbox(
    "Choose Cost",
    le_biaya.classes_
)

# ======================
# Encode input
# ======================
X_input = np.array([[
    le_diagnosa.transform([diagnosa_input])[0],
    le_institusi.transform([institusi_input])[0],
    le_area.transform([area_input])[0],
    le_tipe.transform([tipe_input])[0],
    le_biaya.transform([biaya_input])[0],
]])

# ======================
# Prediction
# ======================
if st.button("Prediksi"):
    hasil = model.predict(X_input)[0]

    label_map = {
        0: "Pelayanan kelas bawah",
        1: "Pelayanan kelas atas",
        2: "Pelayanan kelas menengah"
    }

    st.success(f"Predict Result: {label_map.get(hasil, 'Unknown')}")

# ======================
# Footer
# ======================
st.markdown("""
<div class="footer">
    <a href="https://github.com/ryanputranda/Prediksi-Layanan-Data-EMR-dengan-Random-Forest-dan-GridSearchCV" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
    </a>
    Ryan Portfolio
</div>
""", unsafe_allow_html=True)
