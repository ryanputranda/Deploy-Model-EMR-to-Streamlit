import streamlit as st
import joblib
import numpy as np

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Aplikasi Segmentasi Layanan Kesehatan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Session State (Navigation)
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# =====================================================
# CUSTOM CSS (MODERN UI)
# =====================================================
st.markdown("""
<style>

/* ===== Global ===== */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #0e1117;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #9ba1a6;
    margin-bottom: 16px;
}

/* Sidebar buttons full width */
div.stButton > button {
    width: 100%;
    text-align: left;
    padding: 10px 14px;
    border-radius: 8px;
    background-color: #1f2933;
    color: white;
    border: none;
    margin-bottom: 6px;
    font-size: 14px;
}

div.stButton > button:hover {
    background-color: #2563eb;
    color: white;
}

/* ===== Footer ===== */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(90deg, #0e1117, #111827);
    color: #e5e7eb;
    text-align: center;
    padding: 10px 0;
    font-size: 13px;
    z-index: 100;
}

.footer img {
    width: 18px;
    height: 18px;
    vertical-align: middle;
    margin-right: 6px;
}

.footer a {
    color: #e5e7eb;
    text-decoration: none;
}

.footer a:hover {
    color: #60a5fa;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">EMR App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">EMR Prediction System</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🏠 Home"):
        st.session_state.page = "Home"

    if st.button("📊 Segmentasi"):
        st.session_state.page = "Segmentasi"

    if st.button("👨‍💻 Developer"):
        st.session_state.page = "Developer"

# =====================================================
# MAIN CONTENT
# =====================================================
page = st.session_state.page

# ======================
# HOME
# ======================
if page == "Home":
    st.title("📊 Segmentasi Layanan Kesehatan Karyawan")

    st.write("""
    Aplikasi ini digunakan untuk **segmentasi layanan kesehatan karyawan**
    berdasarkan **Electronic Medical Record (EMR)** menggunakan
    **Random Forest Classifier** yang telah dioptimasi dengan **GridSearchCV**.
    """)

    st.markdown("""
    ### 🔎 Model Pipeline
    - Data Cleaning
    - Exploratory Data Analysis (EDA)
    - Clustering
    - Class Balancing
    - Hyperparameter Tuning
    - Classification
    - Evaluation Metrics
    - Model Deployment
    """)

# ======================
# EMR PREDICT
# ======================
elif page == "Segmentasi":
    st.title("📊 Input data masukan anda..")

    model = joblib.load("random_forest_model.pkl")

    le_diagnosa   = joblib.load("le_diagnosa.pkl")
    le_institusi  = joblib.load("le_institusi.pkl")
    le_area       = joblib.load("le_area.pkl")
    le_tipe       = joblib.load("le_tipe.pkl")
    le_biaya      = joblib.load("le_biaya.pkl")

    col1, col2 = st.columns(2)

    with col1:
        diagnosa_input = st.selectbox("Diagnosa", le_diagnosa.classes_)
        institusi_input = st.selectbox("Institusi", le_institusi.classes_)
        area_input = st.selectbox("Area", le_area.classes_)

    with col2:
        tipe_input = st.selectbox("Tipe Layanan", le_tipe.classes_)
        biaya_input = st.selectbox("Besaran Biaya", le_biaya.classes_)

    X_input = np.array([[ 
        le_diagnosa.transform([diagnosa_input])[0],
        le_institusi.transform([institusi_input])[0],
        le_area.transform([area_input])[0],
        le_tipe.transform([tipe_input])[0],
        le_biaya.transform([biaya_input])[0],
    ]])

    if st.button("🔍 Proses Segmentasi"):
        hasil = model.predict(X_input)[0]

        label_map = {
            0: "Pelayanan Kelas Bawah",
            1: "Pelayanan Kelas Atas",
            2: "Pelayanan Kelas Menengah"
        }

        st.success(f"✅ Hasil Segmentasi: **{label_map.get(hasil)}**")

# ======================
# ABOUT
# ======================
elif page == "Developer":
    st.title("👨‍💻 Developer")

    st.write("""
    Aplikasi ini dikembangkan sebagai implementasi **Machine Learning**
    pada data **Electronic Medical Record (EMR)** untuk mendukung
    pengambilan keputusan layanan kesehatan karyawan.
    """)

    st.markdown("""
    **Developer Info**
    - **Nama:** Ryan Putranda Kristianto
    - **Linkedin:** Ryan Putranda Kristianto
    - **Github:** Ryan Putranda
    """)

    st.markdown(
        "🔗 [GitHub Repository](https://github.com/ryanputranda/Prediksi-Layanan-Data-EMR-dengan-Random-Forest-dan-GridSearchCV)"
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <a href="https://github.com/ryanputranda/Prediksi-Layanan-Data-EMR-dengan-Random-Forest-dan-GridSearchCV" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
        Ryan Putranda · Machine Learning Portfolio
    </a>
</div>
""", unsafe_allow_html=True)
