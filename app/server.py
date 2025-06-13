from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil
import os

# Load model dari lokal folder
MODEL_PATH = "app/best_model_efficientnetb7.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model .h5 tidak ditemukan di folder /model")

model = load_model(MODEL_PATH)

# Daftar class
class_names = np.array(['metal', 'battery', 'plastic', 'shoes', 'paper', 'cardboard', 'glass', 'biological'])

# Inisialisasi FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Sampah Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Simpan file temporer
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Preprocessing gambar
        img = image.load_img(temp_path, target_size=(224, 224))  # sesuaikan dengan input model
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]

        return {"predicted_class": pred_class, "confidence": float(np.max(prediction))}
    
    finally:
        os.remove(temp_path)
