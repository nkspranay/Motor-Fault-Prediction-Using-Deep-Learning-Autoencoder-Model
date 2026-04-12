"""
api_server.py
==============
FastAPI server for the motor fault detection dashboard.
Reads from ESP32 via serial, runs autoencoder inference,
and serves real-time data + fault history to the frontend.

Usage:
    uvicorn api_server:app --reload --port 8000
"""

import os
import csv
import threading
import random
import numpy as np
import joblib
import torch
from torch import nn
from collections import deque
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Tuple, List  

# ─────────────── CONFIG ───────────────────
PORT        = "COM5"
BAUD        = 115200
WINDOW_SIZE = 20
N_FEATURES  = 6
INPUT_DIM   = WINDOW_SIZE * N_FEATURES
HISTORY_MAX = 100
TREND_WINDOW = 20
TREND_THRESHOLD = 0.6

FEATURES = ["Voltage", "Current", "Power", "Temperature", "Humidity", "Vibration"]

# ─────────────── THRESHOLDS ───────────────
THRESHOLD         = 0.0
WARNING_THRESHOLD = 0.0
feature_thresholds = np.zeros(N_FEATURES)

# ─────────────── APP ──────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────── MODEL ────────────────────
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),        nn.ReLU(),
            nn.Linear(32, 64),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ─────────────── PATHS ────────────────────
BASE_DIR               = os.path.dirname(os.path.abspath(__file__))
model_path             = os.path.join(BASE_DIR, "model.pth")
scaler_path            = os.path.join(BASE_DIR, "scaler.pkl")
threshold_path         = os.path.join(BASE_DIR, "threshold.npy")
warning_threshold_path = os.path.join(BASE_DIR, "warning_threshold.npy")
feature_thresh_path    = os.path.join(BASE_DIR, "feature_thresholds.npy")

# ─────────────── LOAD MODEL ───────────────
model_loaded = False
model = AutoEncoder(INPUT_DIM)

if all(os.path.exists(p) for p in [
    model_path, scaler_path, threshold_path,
    warning_threshold_path, feature_thresh_path
]):
    print("✔ REAL MODE — loading model files")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    scaler             = joblib.load(scaler_path)
    THRESHOLD          = float(np.load(threshold_path))
    WARNING_THRESHOLD  = float(np.load(warning_threshold_path))
    feature_thresholds = np.load(feature_thresh_path)
    model_loaded       = True
    print(f"  Fault threshold   : {THRESHOLD:.6f}")
    print(f"  Warning threshold : {WARNING_THRESHOLD:.6f}")
else:
    print("⚠ DUMMY MODE — model files not found, using simulated data")

# ─────────────── SERIAL ───────────────────
if model_loaded:
    import serial
    ser    = serial.Serial(PORT, BAUD, timeout=1)
    buffer = deque(maxlen=WINDOW_SIZE)

# ─────────────── SHARED STATE + LOCK ──────
_lock = threading.Lock()

_state = {
    "values"        : {f: 0.0 for f in FEATURES},
    "status"        : "Normal",
    "faults"        : [],
    "prediction"    : False,    # True = pattern suggests upcoming fault
    "mse"           : 0.0,
    "mse_history"   : [],       # Last N MSE values (for frontend chart)
    "history"       : [],       # Fault/warning event log
}

mse_deque = deque(maxlen=100)   # Internal MSE window for trend detection


def get_state() -> dict:
    with _lock:
        return {
            "values"     : dict(_state["values"]),
            "status"     : _state["status"],
            "faults"     : list(_state["faults"]),
            "prediction" : _state["prediction"],
            "mse"        : _state["mse"],
            "mse_history": list(_state["mse_history"]),
            "history"    : list(_state["history"]),
        }


def set_state(**kwargs):
    with _lock:
        for k, v in kwargs.items():
            _state[k] = v


# ─────────────── TREND ────────────────────
def detect_rising_trend() -> Tuple[bool, float]:
    if len(mse_deque) < TREND_WINDOW:
        return False, 0.0
    values = np.array(list(mse_deque)[-TREND_WINDOW:])
    if values.std() < 1e-10:
        return False, 0.0
    x    = np.arange(len(values), dtype=float)
    corr = np.corrcoef(x, values)[0, 1]
    slope = np.polyfit(x, values, 1)[0]
    return corr > TREND_THRESHOLD and slope > 0, slope


# ─────────────── DUMMY ────────────────────
_dummy_temp   = 35.0
_dummy_step   = 0

def generate_dummy() -> Tuple[dict, str, List, bool]:
    global _dummy_temp, _dummy_step
    _dummy_step += 1

    # Slowly rising temperature to simulate a developing fault
    _dummy_temp += random.uniform(-0.1, 0.15)
    _dummy_temp  = max(30, min(75, _dummy_temp))

    current = random.uniform(0.3, 1.5)
    voltage = random.uniform(220, 240)
    values  = {
        "Voltage"    : round(voltage, 2),
        "Current"    : round(current, 4),
        "Power"      : round(voltage * current, 2),
        "Temperature": round(_dummy_temp, 1),
        "Humidity"   : round(random.uniform(50, 70), 1),
        "Vibration"  : round(random.uniform(500, 2000), 0),
    }

    if _dummy_temp > 72:
        return values, "Fault", ["Temperature"], False
    elif _dummy_temp > 68:
        return values, "Warning", [], False
    elif _dummy_temp > 62:
        return values, "Normal", [], True    # prediction: temp trending up
    else:
        return values, "Normal", [], False


# ─────────────── LOG ──────────────────────
_log_path = os.path.join(BASE_DIR, "logs.csv")
_log_file = open(_log_path, "a", newline="")   # Open once — not per-loop
_log_writer = csv.writer(_log_file)


def log_event(status: str, faults: list, mse: float):
    if status != "Normal":
        _log_writer.writerow([datetime.now().isoformat(), status, ";".join(faults), round(mse, 6)])
        _log_file.flush()


# ─────────────── DETECTION LOOP ───────────
anomaly_counter = 0

def detection_loop():
    global anomaly_counter

    while True:
        try:
            if not model_loaded:
                # ── DUMMY MODE ──
                import time
                time.sleep(0.5)
                vals, status, faults, prediction = generate_dummy()

                with _lock:
                    _state["values"]     = vals
                    _state["status"]     = status
                    _state["faults"]     = faults
                    _state["prediction"] = prediction
                    _state["mse"]        = 0.0

                    if status != "Normal" or prediction:
                        _state["history"].append({
                            "time"      : datetime.now().isoformat(timespec="seconds"),
                            "status"    : status,
                            "faults"    : faults,
                            "prediction": prediction,
                        })
                        _state["history"] = _state["history"][-HISTORY_MAX:]

            else:
                # ── REAL MODE ──
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw or raw.startswith("#"):
                    continue

                parts = raw.split(",")
                if len(parts) != N_FEATURES:
                    continue

                try:
                    vals_arr = np.array([float(v) for v in parts])
                except ValueError:
                    continue

                vals_scaled = scaler.transform([vals_arr])[0]
                buffer.append(vals_scaled)

                if len(buffer) < WINDOW_SIZE:
                    continue

                x = torch.tensor(
                    np.array(buffer).reshape(1, -1),
                    dtype=torch.float32
                )

                with torch.no_grad():
                    recon       = model(x)
                    loss        = torch.mean((recon - x) ** 2).item()
                    feat_errors = (recon - x).numpy().reshape(WINDOW_SIZE, N_FEATURES).mean(axis=0)

                mse_deque.append(loss)
                is_rising, _ = detect_rising_trend()

                # Anomaly counter
                if loss > THRESHOLD:
                    anomaly_counter += 1
                else:
                    anomaly_counter = 0

                # Status
                faults = []
                if anomaly_counter >= 3:
                    status = "Fault"
                    for i, fe in enumerate(feat_errors):
                        if fe > feature_thresholds[i]:
                            faults.append(FEATURES[i])
                elif loss > WARNING_THRESHOLD:
                    status = "Warning"
                else:
                    status = "Normal"

                prediction = is_rising and status == "Normal" and loss > WARNING_THRESHOLD * 0.7

                vals_dict = {
                    FEATURES[i]: round(float(vals_arr[i]), 3)
                    for i in range(N_FEATURES)
                }

                with _lock:
                    _state["values"]     = vals_dict
                    _state["status"]     = status
                    _state["faults"]     = faults
                    _state["prediction"] = prediction
                    _state["mse"]        = round(loss, 6)
                    _state["mse_history"].append(round(loss, 6))
                    if len(_state["mse_history"]) > 60:
                        _state["mse_history"].pop(0)

                    if status != "Normal" or prediction:
                        _state["history"].append({
                            "time"      : datetime.now().isoformat(timespec="seconds"),
                            "status"    : status,
                            "faults"    : faults,
                            "prediction": prediction,
                        })
                        _state["history"] = _state["history"][-HISTORY_MAX:]

                log_event(status, faults, loss)

        except Exception as e:
            print(f"Detection loop error: {e}")
            continue


threading.Thread(target=detection_loop, daemon=True).start()

# ─────────────── ENDPOINTS ────────────────
@app.get("/data")
def get_data():
    return get_state()

@app.get("/history")
def get_history():
    with _lock:
        return {"history": list(_state["history"])}

# Fix the health endpoint:
@app.get("/health")
def health():
    return {
        "model_loaded"   : model_loaded,
        "mode"           : "real" if model_loaded else "dummy",
        "fault_threshold": THRESHOLD if model_loaded else None,
        "warn_threshold" : WARNING_THRESHOLD if model_loaded else None,
    }