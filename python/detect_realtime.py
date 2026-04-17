"""
detect_realtime.py
===================
Real-time motor fault detection using per-state autoencoders.
Reads from ESP32 via serial, detects current-based state,
loads the matching model, and classifies faults using
mean ± 3σ bands plus absolute safety limits.

Usage:
    python detect_realtime.py

Model layout expected:
    python/models/running/  model.pth, scaler.pkl, threshold.npy,
                            warning_threshold.npy, feature_means.npy, feature_stds.npy
    python/models/off/      same set
"""

import os
import torch
import serial
import numpy as np
import joblib
from torch import nn
from collections import deque
import time
from typing import Tuple, List, Optional


# ─────────────── CONFIG ───────────────────
PORT               = "COM7"
BAUD               = 115200
WINDOW_SIZE        = 20
N_FEATURES         = 6
INPUT_DIM          = WINDOW_SIZE * N_FEATURES
CONFIRMATION_COUNT = 3      # MSE fault confirmations before FAULT
WARNING_COUNT      = 3      # sustained warnings before escalating
HISTORY_SIZE       = 50
TREND_WINDOW       = 20
TREND_THRESHOLD    = 0.6
STD_MULTIPLIER     = 3.0    # fault band = mean ± STD_MULTIPLIER × std

FEATURES = ["Voltage", "Current", "Power", "Temperature", "Humidity", "Vibration"]
UNITS    = ["V", "A", "W", "°C", "%", ""]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# ABSOLUTE SAFETY LIMITS
# Hard physical bounds — flagged regardless of MSE or stats.
# Represent electrically/mechanically unsafe values.
# Humidity intentionally excluded — high humidity is operational,
# not a safety risk on its own.
# ─────────────────────────────────────────────────────────────
SAFETY_LIMITS = {
    "Voltage"    : (200.0, 245.0),   # India grid standard
    "Current"    : (0.05,  0.45),
    "Power"      : (10.0,  95.0),
    "Temperature": (20.0,  60.0),
    "Vibration"  : (500.0, 4095.0),
}

# ─────────────────────────────────────────────────────────────
# STATE DETECTION
# Deadband between 0.10–0.28A — hold previous state in this
# range to avoid model thrashing at the boundary.
# ─────────────────────────────────────────────────────────────
STATE_CURRENT_BANDS = {
    "off":     (0.00, 0.10),
    "running": (0.28, 0.42),
}
DEADBAND_LOW        = 0.10
DEADBAND_HIGH       = 0.28
STATE_CONFIRM_COUNT = 5     # consecutive readings before accepting a state switch
TRANSITION_DELTA    = 0.25  # current jump size that triggers a buffer reset


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


# ─────────────── LOAD MODEL FOR A STATE ───
def load_state_model(state: str) -> dict:
    """Load all artifacts for a given state from models/<state>/"""
    model_dir = os.path.join(BASE_DIR, "models", state)

    model = AutoEncoder(INPUT_DIM)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu")
    )
    model.eval()

    return {
        "model"            : model,
        "scaler"           : joblib.load(os.path.join(model_dir, "scaler.pkl")),
        "threshold"        : float(np.load(os.path.join(model_dir, "threshold.npy"))),
        "warning_threshold": float(np.load(os.path.join(model_dir, "warning_threshold.npy"))),
        "feature_means"    : np.load(os.path.join(model_dir, "feature_means.npy")),
        "feature_stds"     : np.load(os.path.join(model_dir, "feature_stds.npy")),
    }


# ─────────────── STATE DETECTION ──────────
def detect_state(current: float) -> Optional[str]:
    """
    Returns state name if current is clearly in a band,
    None if in the deadband (caller holds previous state).
    """
    if DEADBAND_LOW <= current < DEADBAND_HIGH:
        return None

    for state, (lo, hi) in STATE_CURRENT_BANDS.items():
        if lo <= current < hi:
            return state

    return None


# ─────────────── TREND DETECTOR ───────────
def detect_rising_trend(history: deque) -> Tuple[bool, float]:
    if len(history) < TREND_WINDOW:
        return False, 0.0

    values = np.array(list(history)[-TREND_WINDOW:])
    x      = np.arange(len(values), dtype=float)

    if values.std() < 1e-10:
        return False, 0.0

    corr  = np.corrcoef(x, values)[0, 1]
    slope = np.polyfit(x, values, 1)[0]

    return corr > TREND_THRESHOLD and slope > 0, slope


# ─────────────── SAFETY LIMIT CHECK ───────
def check_safety_limits(vals: np.ndarray) -> List[dict]:
    """
    Check absolute physical safety limits every reading.
    Fires before the model — catches dangerous values even
    if MSE happens to be low.
    """
    violations = []
    for i, feature in enumerate(FEATURES):
        if feature not in SAFETY_LIMITS:
            continue
        lo, hi = SAFETY_LIMITS[feature]
        val    = vals[i]
        if val < lo or val > hi:
            violations.append({
                "feature": feature,
                "type"   : "OUT_OF_RANGE",
                "value"  : round(float(val), 3),
            })
    return violations


# ─────────────── FAULT CLASSIFIER ─────────
def classify_faults(vals: np.ndarray,
                    feature_means: np.ndarray,
                    feature_stds: np.ndarray) -> List[dict]:
    """
    Classify which features caused the fault using mean ± STD_MULTIPLIER × std.

    ABNORMAL_HIGH    : value above mean + 2σ
    ABNORMAL_LOW     : value below mean - 2σ
    ABNORMAL_PATTERN : value within normal band — autoencoder caught
                       a relationship anomaly between features.
                       Often the most meaningful signal.
    """
    faults = []
    for i, val in enumerate(vals):
        mean = feature_means[i]
        std  = feature_stds[i]

        if val > mean + STD_MULTIPLIER * std:
            fault_type = "ABNORMAL_HIGH"
        elif val < mean - STD_MULTIPLIER * std:
            fault_type = "ABNORMAL_LOW"
        else:
            fault_type = "ABNORMAL_PATTERN"

        faults.append({
            "feature": FEATURES[i],
            "type"   : fault_type,
            "value"  : round(float(val), 3),
        })

    return faults


# ─────────────── DISPLAY ──────────────────
def print_status(status, loss, vals, state,
                 faults=None, trend=False, slope=0.0):
    colors = {
        "NORMAL" : "\033[92m",
        "WARNING": "\033[93m",
        "FAULT"  : "\033[91m",
        "PREDICT": "\033[95m",
    }
    RESET = "\033[0m"
    color = colors.get(status, "")

    readings = "  ".join(
        f"{FEATURES[i]}={vals[i]:.2f}{UNITS[i]}"
        for i in range(N_FEATURES)
    )

    print(f"\n{color}[{status}]{RESET}  state={state}  MSE={loss:.6f}  {readings}")

    if trend and status == "NORMAL":
        print(f"  \033[95m⚑ PREDICTION: rising error (slope={slope:.2e}) — possible fault developing\033[0m")

    if faults:
        print(f"  Fault details:")
        for f in faults:
            print(f"    → {f['feature']:12s} [{f['type']}]  value={f['value']}")


# ─────────────── RESET HELPER ─────────────
def reset_counters():
    """Full counter reset — on transition, state switch, or confirmed fault."""
    global anomaly_counter, warning_counter, safety_counters
    anomaly_counter = 0
    warning_counter = 0
    safety_counters = {f: 0 for f in SAFETY_LIMITS}


# ─────────────── INIT ─────────────────────
print("Loading models...")
state_models = {}
for state in STATE_CURRENT_BANDS:
    try:
        state_models[state] = load_state_model(state)
        cfg = state_models[state]
        print(f"  ✔ {state:10s}  fault={cfg['threshold']:.6f}  "
              f"warning={cfg['warning_threshold']:.6f}")
        print(f"             means : {np.round(cfg['feature_means'], 3)}")
        print(f"             stds  : {np.round(cfg['feature_stds'], 3)}")
    except FileNotFoundError:
        print(f"  ✘ {state}: model files not found in models/{state}/ — skipping")

if not state_models:
    raise SystemExit("No models loaded. Run train_model.py first.")


# ─────────────── STATE ────────────────────
current_state     = "unknown"
pending_state     = None
pending_state_ctr = 0
buffer            = deque(maxlen=WINDOW_SIZE)
mse_history       = deque(maxlen=HISTORY_SIZE)
active            = None   # currently loaded model artifacts

anomaly_counter   = 0
warning_counter   = 0
safety_counters   = {f: 0 for f in SAFETY_LIMITS}
fault_active      = False


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

print(f"{'─' * 60}")
print(f"  Motor Fault Detection — Real-time (per-state models)")
print(f"  STD multiplier    : ±{STD_MULTIPLIER}σ")
print(f"  Ctrl+C to stop")
print(f"{'─' * 60}")


# ─────────────── MAIN LOOP ────────────────
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

        # ── Hard reject on obviously corrupt voltage readings ──
        if not (180.0 <= vals[0] <= 260.0):
            continue

        current_reading = vals[1]

        # ── State detection with deadband + confirmation ──
        detected = detect_state(current_reading)

        if detected is not None and detected != current_state:
            if detected == pending_state:
                pending_state_ctr += 1
            else:
                pending_state     = detected
                pending_state_ctr = 1

            if pending_state_ctr >= STATE_CONFIRM_COUNT:
                if pending_state in state_models:
                    current_state = pending_state
                    active        = state_models[current_state]
                    buffer.clear()
                    mse_history.clear()
                    reset_counters()
                    print(f"\n  ── State switched to: {current_state.upper()} ──")
                    print(f"     fault threshold   : {active['threshold']:.6f}")
                    print(f"     warning threshold : {active['warning_threshold']:.6f}")
                pending_state     = None
                pending_state_ctr = 0
        elif detected == current_state:
            pending_state     = None
            pending_state_ctr = 0

        # ── Skip if no valid model loaded yet ──
        if active is None:
            print(f"  Waiting for known state... (current={current_reading:.3f}A)", end="\r")
            continue

        # ── Safety limit check — runs every reading, before model ──
        safety_violations = check_safety_limits(vals)
        for sv in safety_violations:
            safety_counters[sv["feature"]] += 1
        for feat in list(safety_counters):
            if feat not in [sv["feature"] for sv in safety_violations]:
                safety_counters[feat] = 0

        # Only escalate safety violations that have been sustained
        sustained_safety_faults = [
            sv for sv in safety_violations
            if safety_counters[sv["feature"]] >= WARNING_COUNT
        ]

        # ── Transition detection — large current jump resets buffer ──
        if len(buffer) > 0:
            prev_vals = active["scaler"].inverse_transform([buffer[-1]])[0]
            if abs(current_reading - prev_vals[1]) > TRANSITION_DELTA:
                buffer.clear()
                mse_history.clear()
                reset_counters()
                print("↺ Transition detected → resetting buffer")
                continue

        # ── Scale and clip ──
        vals_scaled = active["scaler"].transform([vals])[0]
        vals_scaled = np.clip(vals_scaled, 0.0, 1.0)
        buffer.append(vals_scaled)

        if len(buffer) < WINDOW_SIZE:
            print(f"  Buffering... {len(buffer)}/{WINDOW_SIZE}", end="\r")
            continue

        # ── Model inference ──
        x = torch.tensor(
            np.array(buffer).reshape(1, -1), dtype=torch.float32
        )
        with torch.no_grad():
            recon = active["model"](x)
            loss  = torch.mean((recon - x) ** 2).item()

        mse_history.append(loss)

        # Need enough history before trend detection is meaningful
        if len(mse_history) < 10:
            print_status("NORMAL", loss, vals, current_state)
            continue

        is_rising, slope = detect_rising_trend(mse_history)

        # ── Anomaly counter ──
        if loss > active["threshold"] or sustained_safety_faults:
            anomaly_counter += 1
        else:
            anomaly_counter = 0

        # ── Status decision ──
        if anomaly_counter >= CONFIRMATION_COUNT:
            fault_active = True

        if fault_active:
            anomaly_counter = CONFIRMATION_COUNT

            faults = classify_faults(
                vals,
                active["feature_means"],
                active["feature_stds"],
            )
            faults.extend(sustained_safety_faults)

            print_status("FAULT", loss, vals, current_state, faults=faults)

            # Exit FAULT only when fully normal
            if loss <= active["warning_threshold"] and not sustained_safety_faults:
                fault_active = False
                anomaly_counter = 0
                warning_counter = 0
                mse_history.clear()
                reset_counters()

        else:
            if loss > active["warning_threshold"] or sustained_safety_faults:
                warning_counter += 1
            else:
                warning_counter = 0

            if warning_counter >= WARNING_COUNT:
                warnings = []
                if loss > active["warning_threshold"]:
                    warnings.append({
                        "feature": "System",
                        "type"   : "PATTERN_DEVIATION",
                        "value"  : round(loss, 6),
                    })
                warnings.extend(sustained_safety_faults)
                print_status("WARNING", loss, vals, current_state, faults=warnings)

            elif is_rising and loss > active["warning_threshold"] * 0.7:
                print_status("PREDICT", loss, vals, current_state,
                             trend=True, slope=slope)

            else:
                print_status("NORMAL", loss, vals, current_state,
                             trend=is_rising, slope=slope)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
    ser.close()