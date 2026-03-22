from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import threading
import torch
import serial
import numpy as np
import joblib
from torch import nn
from collections import deque
import csv
from datetime import datetime
import os
import random

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WINDOW_SIZE = 20

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model.pth")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
threshold_path = os.path.join(BASE_DIR, "threshold.npy")
feature_thresholds_path = os.path.join(BASE_DIR, "feature_thresholds.npy")

# ---------------- MODEL ----------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW_SIZE * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, WINDOW_SIZE * 5)
        )

    def forward(self, x):
        return self.net(x)

model = AutoEncoder()

# ---------------- LOAD ----------------
model_loaded = False

if (
    os.path.exists(model_path) and
    os.path.exists(scaler_path) and
    os.path.exists(threshold_path) and
    os.path.exists(feature_thresholds_path)
):
    print("✔ REAL MODE")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scaler = joblib.load(scaler_path)
    THRESHOLD = float(np.load(threshold_path))
    feature_thresholds = np.load(feature_thresholds_path)

    WARNING_THRESHOLD = 0.7 * THRESHOLD
    model_loaded = True

else:
    print("⚠ DUMMY MODE")

feature_names = ["Voltage", "Current", "Power", "Temperature", "Vibration"]

# ---------------- SERIAL ----------------
if model_loaded:
    ser = serial.Serial("COM5", 115200, timeout=1)
    buffer = deque(maxlen=WINDOW_SIZE)

# ---------------- DATA ----------------
latest_data = {
    "values": {},
    "status": "Normal",
    "faults": [],
    "history": []
}

# ---------------- DUMMY ----------------
def generate_dummy():
    temp = random.uniform(40, 75)

    values = {
        "Voltage": random.uniform(210, 240),
        "Current": random.uniform(0.5, 2),
        "Power": random.uniform(80, 200),
        "Temperature": temp,
        "Vibration": random.uniform(0.1, 1)
    }

    # realistic warning/fault logic
    if temp > 72:
        return values, "Fault", ["Temperature"]
    elif temp > 68:
        return values, "Warning", []
    else:
        return values, "Normal", []

# ---------------- LOOP ----------------
def detection_loop():
    global latest_data

    while True:
        try:

            # -------- DUMMY MODE --------
            if not model_loaded:
                vals, status, faults = generate_dummy()

                latest_data["values"] = vals
                latest_data["status"] = status
                latest_data["faults"] = faults

            # -------- REAL MODE --------
            else:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue

                parts = raw.split(",")
                if len(parts) != 5:
                    continue

                vals = np.array([float(v) for v in parts])
                vals_scaled = scaler.transform([vals])[0]

                buffer.append(vals_scaled)

                if len(buffer) < WINDOW_SIZE:
                    continue

                x = torch.tensor(np.array(buffer).reshape(1, -1), dtype=torch.float32)

                with torch.no_grad():
                    recon = model(x)
                    loss = torch.mean((recon - x) ** 2).item()

                    error = (recon - x).numpy().reshape(WINDOW_SIZE, 5)
                    feature_errors = error.mean(axis=0)

                latest_data["values"] = {
                    "Voltage": float(vals[0]),
                    "Current": float(vals[1]),
                    "Power": float(vals[2]),
                    "Temperature": float(vals[3]),
                    "Vibration": float(vals[4]),
                }

                status = "Normal"
                faults = []

                if loss > THRESHOLD:
                    status = "Fault"
                    for i, fe in enumerate(feature_errors):
                        if fe > feature_thresholds[i]:
                            faults.append(feature_names[i])

                elif loss > WARNING_THRESHOLD:
                    status = "Warning"

                latest_data["status"] = status
                latest_data["faults"] = faults

                with open("logs.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now(),
                        status,
                        ",".join(faults),
                        loss
                    ])

            # -------- HISTORY --------
            if latest_data["status"] != "Normal":
                latest_data["history"].append({
                    "time": str(datetime.now()),
                    "faults": latest_data["faults"]
                })
                latest_data["history"] = latest_data["history"][-50:]

        except Exception as e:
            print("Error:", e)
            continue

threading.Thread(target=detection_loop, daemon=True).start()

@app.get("/data")
def get_data():
    return latest_data