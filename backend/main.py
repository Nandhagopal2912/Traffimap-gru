import numpy as np
import h5py
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# Enable CORS so the browser doesn't block your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Initialization ---
try:
    with h5py.File("metr-la.h5", "r") as f:
        data = f['df']['block0_values'][:]
    
    data = np.nan_to_num(data)
    # Calculate global mean and std for MPH conversion later
    mean_val = float(data.mean())
    std_val = float(data.std())
    
    # Normalize the data for the model
    data = (data - mean_val) / std_val

    # Load sensor metadata
    df_coords = pd.read_csv("graph_sensor_locations.csv").sort_values("sensor_id")
    sensor_ids = df_coords['sensor_id'].tolist()
    sensor_coords = df_coords[['latitude', 'longitude']].values.tolist()

    # Load the GRU model
    model = load_model("metr_gru_model.keras")
except Exception as e:
    print(f"CRITICAL ERROR: Check your file paths! {e}")

current_idx = 12

# --- API Endpoints ---

@app.get("/sensors")
def get_sensors():
    return {
        "sensors": sensor_coords, 
        "ids": sensor_ids,
        "mean": mean_val,
        "std": std_val
    }

@app.get("/predict")
def predict():
    global current_idx
    # Loop back to the start if we reach the end of the dataset
    if current_idx >= len(data): 
        current_idx = 12

    # Get the last 12 time steps for the prediction
    seq = data[current_idx-12:current_idx]
    x = np.expand_dims(seq, axis=0)
    
    # Model inference
    pred = model.predict(x)[0].tolist()
    
    # Calculate simple analytics
    avg_speed = sum(pred) / len(pred)
    indexed_preds = [{"id": sensor_ids[i], "val": v} for i, v in enumerate(pred)]
    top_congested = sorted(indexed_preds, key=lambda x: x['val'])[:5]

    current_idx += 1
    return {
        "prediction": pred,
        "metrics": {
            "average": round(avg_speed, 3),
            "top_congested": top_congested
        }
    }