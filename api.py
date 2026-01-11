from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

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
# FastAPI init
# ======================
app = FastAPI(
    title="EMR Segmentation API",
    description="API untuk segmentasi layanan kesehatan karyawan berbasis EMR menggunakan Random Forest",
    version="1.0"
)

# ======================
# Request Schema
# ======================
class EMRRequest(BaseModel):
    diagnosa: str
    institusi: str
    area: str
    tipe: str
    biaya: str

# ======================
# Response Schema
# ======================
class EMRResponse(BaseModel):
    segmentasi: str

# ======================
# Root Endpoint
# ======================
@app.get("/")
def root():
    return {
        "message": "EMR Segmentation API is running",
        "docs": "/docs"
    }

# ======================
# Prediction Endpoint
# ======================
@app.post("/predict", response_model=EMRResponse)
def predict_emr(data: EMRRequest):

    X_input = np.array([[
        le_diagnosa.transform([data.diagnosa])[0],
        le_institusi.transform([data.institusi])[0],
        le_area.transform([data.area])[0],
        le_tipe.transform([data.tipe])[0],
        le_biaya.transform([data.biaya])[0],
    ]])

    hasil = model.predict(X_input)[0]

    label_map = {
        0: "Pelayanan kelas bawah",
        1: "Pelayanan kelas atas",
        2: "Pelayanan kelas menengah"
    }

    return {
        "segmentasi": label_map.get(hasil, "Unknown")
    }
