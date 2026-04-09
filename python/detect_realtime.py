"""
detect_realtime.py
===================
Real-time motor fault detection using the trained autoencoder.
Reads from ESP32 via serial, detects anomalies, and predicts
upcoming faults from repeating reconstruction error patterns.

Usage:
    python detect_realtime.py
"""

import torch
import serial
import numpy as np
import joblib
from torch import nn
from collections import deque
import time

# ─────────────── CONFIG ───────────────────
PORT              = "COM5"
BAUD              = 115200
WINDOW_SIZE       = 20
N_FEATURES        = 6
INPUT_DIM         = WINDOW_SIZE * N_FEATURES
CONFIRMATION_COUNT = 3       # Consecutive anomalies before confirmed fault
HISTORY_SIZE      = 50       # MSE history for trend detection
TREND_WINDOW      = 20       # Sliding window for trend analysis
TREND_THRESHOLD   = 0.6      # Pearson r for "rising trend" detection

FEATURES = ["Voltage", "Current", "Power", "Temperature", "Humidity", "Vibration"]

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


# ─────────────── LOAD FILES ───────────────
print("Loading model...")
model = AutoEncoder(INPUT_DIM)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

scaler             = joblib.load("scaler.pkl")
THRESHOLD          = float(np.load("threshold.npy"))
WARNING_THRESHOLD  = float(np.load("warning_threshold.npy"))
feature_thresholds = np.load("feature_thresholds.npy")

feature_mins = scaler.data_min_
feature_maxs = scaler.data_max_

print(f"  Fault threshold   : {THRESHOLD:.6f}")
print(f"  Warning threshold : {WARNING_THRESHOLD:.6f}")
print(f"  Feature thresholds: {feature_thresholds}")

# ─────────────── STATE ────────────────────
buffer          = deque(maxlen=WINDOW_SIZE)
anomaly_counter = 0
mse_history     = deque(maxlen=HISTORY_SIZE)   # For trend detection

# ─────────────── TREND DETECTOR ───────────
def detect_rising_trend(history: deque) -> tuple[bool, float]:
    """
    Returns (is_rising, slope) based on linear regression of recent MSE values.
    A consistently rising reconstruction error before threshold is crossed
    is an early warning of an upcoming fault.
    """
    if len(history) < TREND_WINDOW:
        return False, 0.0

    values = np.array(list(history)[-TREND_WINDOW:])
    x      = np.arange(len(values), dtype=float)

    # Pearson correlation as trend strength
    if values.std() < 1e-10:
        return False, 0.0

    corr   = np.corrcoef(x, values)[0, 1]
    slope  = np.polyfit(x, values, 1)[0]

    is_rising = corr > TREND_THRESHOLD and slope > 0
    return is_rising, slope


# ─────────────── FAULT CLASSIFIER ─────────
def classify_faults(feature_errors: np.ndarray, vals_original: np.ndarray) -> list[dict]:
    """
    For each feature exceeding its threshold, classify the fault type
    using actual physical values (not scaled).
    """
    faults = []
    for i, fe in enumerate(feature_errors):
        if fe > feature_thresholds[i]:
            val      = vals_original[i]
            f_min    = feature_mins[i]
            f_max    = feature_maxs[i]
            midpoint = (f_min + f_max) / 2

            # Classify based on position within training range
            if val > f_max * 1.05:
                fault_type = "HIGH"
            elif val < f_min * 0.95:
                fault_type = "LOW"
            elif val > midpoint:
                fault_type = "ABNORMAL_HIGH"
            else:
                fault_type = "ABNORMAL_LOW"

            faults.append({
                "feature"   : FEATURES[i],
                "type"      : fault_type,
                "value"     : round(float(val), 3),
                "error"     : round(float(fe), 6),
            })
    return faults


# ─────────────── DISPLAY ──────────────────
UNITS = ["V", "A", "W", "°C", "%", ""]

def print_status(status, loss, vals, faults=None, trend=False, slope=0.0):
    status_color = {
        "NORMAL" : "\033[92m",   # green
        "WARNING": "\033[93m",   # yellow
        "FAULT"  : "\033[91m",   # red
        "PREDICT": "\033[95m",   # magenta
    }.get(status, "")
    RESET = "\033[0m"

    readings = "  ".join(
        f"{FEATURES[i]}={vals[i]:.2f}{UNITS[i]}"
        for i in range(N_FEATURES)
    )

    print(f"\n{status_color}[{status}]{RESET}  MSE={loss:.6f}  {readings}")

    if trend and status == "NORMAL":
        print(f"  \033[95m⚑ PREDICTION: Error rising (slope={slope:.2e}) — possible fault developing\033[0m")

    if faults:
        print(f"  Fault details:")
        for f in faults:
            print(f"    → {f['feature']:12s} [{f['type']}]  value={f['value']}")


# ─────────────── SERIAL ───────────────────
print(f"\nConnecting to {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    ser.flushInput()
    print(f"✔ Connected. Starting detection...\n")
except serial.SerialException as e:
    print(f"✘ Could not open {PORT}: {e}")
    raise SystemExit(1)

# ─────────────── MAIN LOOP ────────────────
print(f"{'─' * 60}")
print(f"  Motor Fault Detection — Real-time")
print(f"  Fault threshold   : {THRESHOLD:.6f}")
print(f"  Warning threshold : {WARNING_THRESHOLD:.6f}")
print(f"  Ctrl+C to stop")
print(f"{'─' * 60}")

try:
    while True:
        raw = ser.readline().decode(errors="ignore").strip()

        if not raw or raw.startswith("#"):
            continue

        parts = raw.split(",")
        if len(parts) != N_FEATURES:
            continue

        try:
            vals = np.array([float(v) for v in parts])
        except ValueError:
            continue

        # Scale
        vals_scaled = scaler.transform([vals])[0]
        buffer.append(vals_scaled)

        if len(buffer) < WINDOW_SIZE:
            print(f"  Buffering... {len(buffer)}/{WINDOW_SIZE}", end="\r")
            continue

        # ── Inference ──
        x = torch.tensor(
            np.array(buffer).reshape(1, -1),
            dtype=torch.float32
        )
        with torch.no_grad():
            recon         = model(x)
            loss          = torch.mean((recon - x) ** 2).item()
            feat_errors   = (recon - x).numpy().reshape(WINDOW_SIZE, N_FEATURES).mean(axis=0)

        mse_history.append(loss)

        # ── Trend detection ──
        is_rising, slope = detect_rising_trend(mse_history)

        # ── Anomaly logic ──
        if loss > THRESHOLD:
            anomaly_counter += 1
        else:
            anomaly_counter = 0

        # ── Determine status ──
        if anomaly_counter >= CONFIRMATION_COUNT:
            faults = classify_faults(feat_errors, vals)
            print_status("FAULT", loss, vals, faults=faults)

        elif loss > WARNING_THRESHOLD:
            print_status("WARNING", loss, vals)

        elif is_rising and loss > WARNING_THRESHOLD * 0.7:
            # Error trending up but hasn't crossed warning yet
            print_status("PREDICT", loss, vals, trend=True, slope=slope)

        else:
            print_status("NORMAL", loss, vals, trend=is_rising, slope=slope)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
    ser.close()