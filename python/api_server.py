"""
api_server.py
==============
FastAPI server for the motor fault detection dashboard.
Reads from ESP32 via serial, runs per-state autoencoder inference,
and serves real-time data + fault history to the frontend.

States : running / off  (deadband 0.10–0.28 A holds previous state)

Fault persistence:
    FAULT   → persists until MSE drops below warning_threshold
    WARNING → persists until MSE drops below warning_threshold
    Normal  → default when resolved

Glitch protection:
    - Individual readings rejected if any feature is beyond GLITCH_STD_MULT
      standard deviations from the training mean, preventing corrupt sensor
      readings from contaminating the inference window and spiking MSE.

Usage:
    uvicorn api_server:app --reload --port 8000
"""

import os
import csv
import time
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
from typing import Optional, List, Tuple

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ─────────────── CONFIG ───────────────────
PORT        = "COM5"
BAUD        = 115200
WINDOW_SIZE = 20
N_FEATURES  = 6
INPUT_DIM   = WINDOW_SIZE * N_FEATURES
HISTORY_MAX = 100
TREND_WINDOW    = 20
TREND_THRESHOLD = 0.6
STD_MULTIPLIER  = 3.0
GLITCH_STD_MULT = 6.0   # reject reading if any feature this many stds from mean

FEATURES = ["Voltage", "Current", "Power", "Temperature", "Humidity", "Vibration"]

STATE_BANDS = {
    "off":     (0.00, 0.10),
    "running": (0.28, 0.42),
}
DEADBAND_LOW        = 0.10
DEADBAND_HIGH       = 0.28
STATE_CONFIRM_COUNT = 5
TRANSITION_DELTA    = 0.25

CONFIRMATION_COUNT = 3
WARNING_COUNT      = 3

SAFETY_LIMITS = {
    "Voltage"    : (200.0, 245.0),
    "Current"    : (0.05,  0.42),
    "Power"      : (10.0,  95.0),
    "Temperature": (20.0,  60.0),
    "Vibration"  : (500.0, 4095.0),
}
SAFETY_CONFIRM_COUNT = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


# ─────────────── LOAD MODELS ──────────────
def load_state_model(state: str) -> Optional[dict]:
    model_dir = os.path.join(BASE_DIR, "models", state)
    required  = ["model.pth", "scaler.pkl", "threshold.npy",
                 "warning_threshold.npy", "feature_means.npy", "feature_stds.npy"]
    if any(not os.path.exists(os.path.join(model_dir, f)) for f in required):
        return None
    m = AutoEncoder(INPUT_DIM)
    m.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu")
    )
    m.eval()
    return {
        "model"            : m,
        "scaler"           : joblib.load(os.path.join(model_dir, "scaler.pkl")),
        "threshold"        : float(np.load(os.path.join(model_dir, "threshold.npy"))),
        "warning_threshold": float(np.load(os.path.join(model_dir, "warning_threshold.npy"))),
        "feature_means"    : np.load(os.path.join(model_dir, "feature_means.npy")),
        "feature_stds"     : np.load(os.path.join(model_dir, "feature_stds.npy")),
    }


print("Loading models...")
state_models: dict = {}
for _s in STATE_BANDS:
    cfg = load_state_model(_s)
    if cfg:
        state_models[_s] = cfg
        print(f"  ✔ {_s:10s}  fault={cfg['threshold']:.6f}  "
              f"warning={cfg['warning_threshold']:.6f}")
    else:
        print(f"  ✘ {_s}: model files missing in models/{_s}/")

model_loaded = bool(state_models)
print("✔ REAL MODE" if model_loaded else "⚠ DUMMY MODE — using simulated data")


# ─────────────── APP ──────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ─────────────── SHARED STATE ─────────────
_lock  = threading.Lock()
_state = {
    "values"     : {f: 0.0 for f in FEATURES},
    "status"     : "Normal",
    "motor_state": "unknown",
    "faults"     : [],
    "prediction" : False,
    "mse"        : 0.0,
    "mse_history": deque(maxlen=60),
    "history"    : [],
}


def get_state() -> dict:
    with _lock:
        return {
            "values"     : dict(_state["values"]),
            "status"     : _state["status"],
            "motor_state": _state["motor_state"],
            "faults"     : list(_state["faults"]),
            "prediction" : _state["prediction"],
            "mse"        : _state["mse"],
            "mse_history": list(_state["mse_history"]),
            "history"    : list(_state["history"]),
        }


# ─────────────── HELPERS ──────────────────
def detect_rising_trend(mse_deque: deque) -> Tuple[bool, float]:
    if len(mse_deque) < TREND_WINDOW:
        return False, 0.0
    values = np.array(list(mse_deque)[-TREND_WINDOW:])
    if values.std() < 1e-10:
        return False, 0.0
    x    = np.arange(len(values), dtype=float)
    corr = np.corrcoef(x, values)[0, 1]
    slope= np.polyfit(x, values, 1)[0]
    return (corr > TREND_THRESHOLD and slope > 0), slope


def is_glitch(vals: np.ndarray, means: np.ndarray, stds: np.ndarray) -> bool:
    for i in range(N_FEATURES):
        if stds[i] > 0 and abs(vals[i] - means[i]) > GLITCH_STD_MULT * stds[i]:
            return True
    return False


def check_safety(vals: np.ndarray, counters: dict) -> List[dict]:
    violations = []
    for i, feat in enumerate(FEATURES):
        if feat not in SAFETY_LIMITS:
            counters[feat] = 0
            continue
        lo, hi = SAFETY_LIMITS[feat]
        if vals[i] < lo or vals[i] > hi:
            counters[feat] = counters.get(feat, 0) + 1
        else:
            counters[feat] = 0
        if counters[feat] >= SAFETY_CONFIRM_COUNT:
            violations.append({"feature": feat, "type": "OUT_OF_RANGE",
                                "value": round(float(vals[i]), 3)})
    return violations


def classify_faults(vals: np.ndarray, means: np.ndarray,
                    stds: np.ndarray) -> List[dict]:
    faults = []
    for i, val in enumerate(vals):
        mean, std = means[i], stds[i]
        if val > mean + STD_MULTIPLIER * std:
            kind = "ABNORMAL_HIGH"
        elif val < mean - STD_MULTIPLIER * std:
            kind = "ABNORMAL_LOW"
        else:
            kind = "ABNORMAL_PATTERN"
        faults.append({"feature": FEATURES[i], "type": kind,
                        "value": round(float(val), 3)})
    return faults


# ─────────────── DUMMY MODE ───────────────
_dummy_temp = 35.0


def generate_dummy():
    global _dummy_temp
    _dummy_temp = max(30.0, min(75.0, _dummy_temp + random.uniform(-0.1, 0.15)))
    current = random.uniform(0.3, 0.42)
    voltage = random.uniform(220.0, 240.0)
    vals    = {
        "Voltage"    : round(voltage, 2),
        "Current"    : round(current, 4),
        "Power"      : round(voltage * current, 2),
        "Temperature": round(_dummy_temp, 1),
        "Humidity"   : round(random.uniform(50.0, 70.0), 1),
        "Vibration"  : round(random.uniform(500.0, 2000.0), 0),
    }
    if _dummy_temp > 72:
        return vals, "Fault", [{"feature": "Temperature",
                                 "type": "ABNORMAL_HIGH", "value": _dummy_temp}], False
    elif _dummy_temp > 68:
        return vals, "Warning", [{"feature": "System",
                                   "type": "PATTERN_DEVIATION", "value": 0.0}], False
    elif _dummy_temp > 62:
        return vals, "Normal", [], True
    else:
        return vals, "Normal", [], False


# ─────────────── LOGGING ──────────────────
_log_path   = os.path.join(BASE_DIR, "logs.csv")
_log_file   = open(_log_path, "a", newline="")
_log_writer = csv.writer(_log_file)


def log_event(status: str, faults: list, mse: float):
    if status != "Normal":
        names = [f["feature"] for f in faults] if faults else []
        _log_writer.writerow([
            datetime.now().isoformat(), status, ";".join(names), round(mse, 6)
        ])
        _log_file.flush()


# ─────────────── DETECTION STATE ──────────
class ServerDetectionState:
    def __init__(self):
        self.current_state  = "unknown"
        self.pending_state  = None
        self.pending_ctr    = 0
        self.buffer         = deque(maxlen=WINDOW_SIZE)
        self.mse_deque      = deque(maxlen=100)
        self.active         = None
        self.anomaly_ctr    = 0
        self.warning_ctr    = 0
        self.safety_ctrs    = {f: 0 for f in FEATURES}
        self.fault_active   = False
        self.warning_active = False

    def reset_counters(self):
        self.anomaly_ctr    = 0
        self.warning_ctr    = 0
        self.fault_active   = False
        self.warning_active = False
        self.safety_ctrs    = {f: 0 for f in FEATURES}

    def switch_state(self, new_state: str):
        self.current_state = new_state
        self.active        = state_models[new_state]
        self.buffer.clear()
        self.mse_deque.clear()
        self.reset_counters()
        print(f"  ── Motor state → {new_state.upper()} ──")


# ─────────────── DETECTION LOOP ───────────
def detection_loop():
    ds  = ServerDetectionState()
    ser = None

    def open_serial():
        nonlocal ser
        if not SERIAL_AVAILABLE:
            return False
        try:
            ser = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            ser.flushInput()
            print(f"✔ Serial connected on {PORT}")
            return True
        except Exception as e:
            print(f"✘ Serial error: {e}")
            return False

    if model_loaded:
        if not open_serial():
            print("⚠ Falling back to dummy mode")

    while True:
        try:
            # ── DUMMY MODE ──
            if not model_loaded or ser is None:
                time.sleep(0.5)
                vals, status, faults, prediction = generate_dummy()
                with _lock:
                    _state["values"]      = vals
                    _state["status"]      = status
                    _state["motor_state"] = "running"
                    _state["faults"]      = faults
                    _state["prediction"]  = prediction
                    _state["mse"]         = 0.0
                    if status != "Normal" or prediction:
                        _state["history"].append({
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "status": status, "faults": faults,
                            "prediction": prediction,
                        })
                        _state["history"] = _state["history"][-HISTORY_MAX:]
                continue

            # ── REAL MODE ──
            try:
                raw = ser.readline().decode(errors="ignore").strip()
            except serial.SerialException:
                print("Serial lost — reconnecting in 3 s...")
                ser = None
                time.sleep(3)
                open_serial()
                continue

            if not raw or raw.startswith("#"):
                continue

            parts = raw.split(",")
            if len(parts) != N_FEATURES:
                continue

            try:
                vals_arr = np.array([float(v) for v in parts])
            except ValueError:
                continue

            if not (180.0 <= vals_arr[0] <= 260.0):
                continue

            current_reading = vals_arr[1]

            # ── Glitch filter ──
            if ds.active is not None:
                if is_glitch(vals_arr, ds.active["feature_means"],
                             ds.active["feature_stds"]):
                    continue

            # ── State detection ──
            detected = None
            if not (DEADBAND_LOW <= current_reading < DEADBAND_HIGH):
                for st, (lo, hi) in STATE_BANDS.items():
                    if lo <= current_reading < hi:
                        detected = st
                        break

            if detected is not None and detected != ds.current_state:
                if detected == ds.pending_state:
                    ds.pending_ctr += 1
                else:
                    ds.pending_state = detected
                    ds.pending_ctr   = 1

                if ds.pending_ctr >= STATE_CONFIRM_COUNT:
                    if detected in state_models:
                        ds.switch_state(detected)
                    ds.pending_state = None
                    ds.pending_ctr   = 0

            elif detected == ds.current_state:
                ds.pending_state = None
                ds.pending_ctr   = 0

            if ds.active is None:
                continue

            # ── Safety ──
            sustained_safety = check_safety(vals_arr, ds.safety_ctrs)

            # ── Transition ──
            if len(ds.buffer) > 0:
                prev_raw = ds.active["scaler"].inverse_transform(
                    np.array(ds.buffer[-1]).reshape(1, -1)
                )[0]
                if abs(current_reading - prev_raw[1]) > TRANSITION_DELTA:
                    ds.buffer.clear()
                    ds.mse_deque.clear()
                    ds.reset_counters()
                    continue

            # ── Scale + buffer ──
            vals_scaled = np.clip(
                ds.active["scaler"].transform([vals_arr])[0], 0.0, 1.0
            )
            ds.buffer.append(vals_scaled)

            if len(ds.buffer) < WINDOW_SIZE:
                continue

            # ── Inference ──
            x = torch.tensor(
                np.array(ds.buffer).reshape(1, -1), dtype=torch.float32
            )
            with torch.no_grad():
                recon = ds.active["model"](x)
                loss  = torch.mean((recon - x) ** 2).item()

            ds.mse_deque.append(loss)
            is_rising, _ = detect_rising_trend(ds.mse_deque)

            thr  = ds.active["threshold"]
            warn = ds.active["warning_threshold"]

            # ── Status machine ──
            fault_trigger   = (loss > thr)  or bool(sustained_safety)
            warning_trigger = (loss > warn) or bool(sustained_safety)

            if fault_trigger:
                ds.anomaly_ctr += 1
            else:
                ds.anomaly_ctr = 0

            if ds.anomaly_ctr >= CONFIRMATION_COUNT:
                ds.fault_active   = True
                ds.warning_active = True

            if ds.fault_active:
                status = "Fault"
                faults = classify_faults(
                    vals_arr, ds.active["feature_means"], ds.active["feature_stds"]
                )
                faults.extend(sustained_safety)
                if loss <= warn and not sustained_safety:
                    ds.fault_active   = False
                    ds.warning_active = False
                    ds.anomaly_ctr    = 0
                    ds.warning_ctr    = 0
                    ds.mse_deque.clear()

            elif ds.warning_active or warning_trigger:
                ds.warning_ctr += 1
                if ds.warning_ctr >= WARNING_COUNT:
                    ds.warning_active = True

                if ds.warning_active:
                    status = "Warning"
                    faults = []
                    if loss > warn:
                        faults.append({"feature": "System",
                                       "type": "PATTERN_DEVIATION",
                                       "value": round(loss, 6)})
                    faults.extend(sustained_safety)
                    if loss <= warn and not sustained_safety:
                        ds.warning_active = False
                        ds.warning_ctr    = 0
                else:
                    status = "Normal"
                    faults = []
            else:
                ds.warning_ctr = 0
                status = "Normal"
                faults = []

            prediction = (
                is_rising and not ds.fault_active
                and not ds.warning_active
                and loss > warn * 0.7
            )

            vals_dict = {FEATURES[i]: round(float(vals_arr[i]), 3)
                         for i in range(N_FEATURES)}

            with _lock:
                _state["values"]      = vals_dict
                _state["status"]      = status
                _state["motor_state"] = ds.current_state
                _state["faults"]      = faults
                _state["prediction"]  = prediction
                _state["mse"]         = round(loss, 6)
                _state["mse_history"].append(round(loss, 6))

                if status != "Normal" or prediction:
                    _state["history"].append({
                        "time"       : datetime.now().isoformat(timespec="seconds"),
                        "status"     : status,
                        "motor_state": ds.current_state,
                        "faults"     : faults,
                        "prediction" : prediction,
                    })
                    _state["history"] = _state["history"][-HISTORY_MAX:]

            log_event(status, faults, loss)

        except Exception as e:
            print(f"Detection loop error: {e}")
            time.sleep(0.1)


threading.Thread(target=detection_loop, daemon=True).start()


# ─────────────── ENDPOINTS ────────────────
@app.get("/data")
def get_data():
    return get_state()


@app.get("/history")
def get_history():
    with _lock:
        return {"history": list(_state["history"])}


@app.get("/health")
def health():
    return {
        "model_loaded" : model_loaded,
        "mode"         : "real" if model_loaded else "dummy",
        "loaded_states": list(state_models.keys()),
        "thresholds"   : {
            s: {"fault"  : state_models[s]["threshold"],
                "warning": state_models[s]["warning_threshold"]}
            for s in state_models
        },
    }