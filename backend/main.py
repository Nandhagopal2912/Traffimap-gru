import numpy as np
import h5py
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset (REAL)
# ✅ dataset
with h5py.File("metr-la.h5", "r") as f:
    data = f['df']['block0_values'][:]


data = np.nan_to_num(data)

# Normalize
mean = data.mean()
std = data.std()
data = (data - mean) / std

#  Load sensor coordinates (REAL)
# File: graph_sensor_locations.csv
df_coords = pd.read_csv("graph_sensor_locations.csv")

# Ensure correct order
df_coords = df_coords.sort_values("sensor_id")

# Extract lat/lon
sensor_coords = df_coords[['latitude', 'longitude']].values.tolist()

#  Load model
model = load_model("metr_lstm_model.keras")

# Time simulation index
current_idx = 12

# ✅ API: get coordinates
@app.get("/sensors")
def get_sensors():
    return {
        "sensors": sensor_coords
    }

# ✅ API: prediction using REAL sequence
@app.get("/predict")
def predict():
    global current_idx

    if current_idx >= len(data):
        current_idx = 12

    seq = data[current_idx-12:current_idx]
    x = np.expand_dims(seq, axis=0)

    pred = model.predict(x)[0]

    current_idx += 1

    return {
        "prediction": pred.tolist()
    }