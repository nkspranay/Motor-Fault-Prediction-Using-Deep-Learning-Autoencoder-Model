"""
detect_realtime.py
===================
Real-time motor fault detection using per-state autoencoders.
Reads from ESP32 via serial, detects current-based state (running / off),
loads the matching model, and classifies faults.

States   : running  (current 0.28–0.42 A)
           off      (current 0.00–0.10 A)
Deadband : 0.10–0.28 A — holds previous state, no model switch

Fault persistence:
    FAULT   → stays FAULT until MSE drops below warning_threshold
    WARNING → stays WARNING until MSE drops below warning_threshold
    NORMAL  → default when neither is active

Glitch protection:
    - Individual readings rejected if any feature is beyond 4×IQR of the
      training distribution (uses feature_means/stds as a proxy).
      This prevents a single corrupt reading (e.g. 0.99V) from contaminating
      the 20-sample window and spiking MSE on normal data.
    - Hard voltage sanity check (180–260V) as first-pass filter.
    - Safety limit violations require SAFETY_CONFIRM_COUNT sustained readings.

Usage:
    python detect_realtime.py
"""

import os
import sys
import time
import torch
import serial
import numpy as np
import joblib
from torch import nn
from collections import deque
from typing import Optional, List, Tuple


# ─────────────── CONFIG ───────────────────
PORT               = "COM7"
BAUD               = 115200
WINDOW_SIZE        = 20
N_FEATURES         = 6
INPUT_DIM          = WINDOW_SIZE * N_FEATURES

CONFIRMATION_COUNT   = 3     # consecutive MSE > fault threshold → FAULT
WARNING_COUNT        = 3     # consecutive MSE > warning threshold → WARNING
HISTORY_SIZE         = 50
TREND_WINDOW         = 20
TREND_THRESHOLD      = 0.6
STD_MULTIPLIER       = 3.0   # for fault classification labels only
GLITCH_STD_MULT      = 8.0   # reject individual reading if any feature is
                              # this many stds from training mean
                              # (higher than fault classifier to avoid over-rejection)

FEATURES = ["Voltage", "Current", "Power", "Temperature", "Humidity", "Vibration"]
UNITS    = ["V", "A", "W", "°C", "%", ""]

STATE_BANDS = {
    "off":     (0.00, 0.10),
    "running": (0.28, 0.42),
}
DEADBAND_LOW        = 0.10
DEADBAND_HIGH       = 0.28
STATE_CONFIRM_COUNT = 3
TRANSITION_DELTA    = 0.25   # current jump that forces a buffer reset

SAFETY_LIMITS = {
    "Voltage"    : (200.0, 245.0),
    "Current"    : (0.05,  0.42),
    "Power"      : (10.0,  95.0),
    "Temperature": (20.0,  60.0),
    "Vibration"  : (2500.0, 4095.0),
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


# ─────────────── LOAD MODEL ───────────────
def load_state_model(state: str) -> dict:
    model_dir = os.path.join(BASE_DIR, "models", state)
    required  = ["model.pth", "scaler.pkl", "threshold.npy",
                 "warning_threshold.npy", "feature_means.npy", "feature_stds.npy"]
    missing   = [f for f in required
                 if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing files for state '{state}' in {model_dir}: {missing}"
        )
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


# ─────────────── GLITCH FILTER ────────────
def is_glitch(vals: np.ndarray, means: np.ndarray, stds: np.ndarray) -> bool:
    """
    Return True if any feature is more than GLITCH_STD_MULT standard deviations
    from the training mean. These are physically impossible readings (e.g. 0.99V
    voltage) that would corrupt the entire 20-sample window's MSE if let through.
    Uses a wider multiplier than the fault classifier to avoid over-rejection.
    """
    for i in range(N_FEATURES):
        if stds[i] > 0 and abs(vals[i] - means[i]) > GLITCH_STD_MULT * stds[i]:
            return True
    return False


# ─────────────── STATE DETECTION ──────────
def detect_state(current: float) -> Optional[str]:
    if DEADBAND_LOW <= current < DEADBAND_HIGH:
        return None
    for state, (lo, hi) in STATE_BANDS.items():
        if lo <= current < hi:
            return state
    return None


# ─────────────── TREND DETECTOR ───────────
def detect_rising_trend(history: deque) -> Tuple[bool, float]:
    if len(history) < TREND_WINDOW:
        return False, 0.0
    values = np.array(list(history)[-TREND_WINDOW:])
    if values.std() < 1e-10:
        return False, 0.0
    x    = np.arange(len(values), dtype=float)
    corr = np.corrcoef(x, values)[0, 1]
    slope= np.polyfit(x, values, 1)[0]
    return (corr > TREND_THRESHOLD and slope > 0), slope


# ─────────────── SAFETY CHECK ─────────────
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


# ─────────────── FAULT CLASSIFIER ─────────
def is_within_band(vals: np.ndarray, means: np.ndarray, stds: np.ndarray, k: float = 3.0) -> bool:
    """
    True if ALL features lie within mean ± k * std
    """
    for i in range(len(vals)):
        if stds[i] > 0:
            if abs(vals[i] - means[i]) > k * stds[i]:
                return False
    return True


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


# ─────────────── DISPLAY ──────────────────
def get_reason(loss, warn, thr, within_band, sustained_safety):
    reasons = []

    # Safety violations (highest priority)
    if sustained_safety:
        for s in sustained_safety:
            reasons.append(f"{s['feature']} out of safe range")

    # Statistical deviation
    if not within_band:
        reasons.append("outside normal band")

    # Model-based deviation
    if loss > thr:
        reasons.append("high reconstruction error")
    elif loss > warn:
        reasons.append("pattern deviation")

    if not reasons:
        return "normal behavior"

    return ", ".join(reasons)

COLORS = {"NORMAL": "\033[92m", "WARNING": "\033[93m",
          "FAULT" : "\033[91m", "PREDICT": "\033[95m"}
RESET  = "\033[0m"


def print_status(status: str, loss: float, vals: np.ndarray, state: str,
                 faults: List[dict] = None, trend: bool = False, slope: float = 0.0,
                 reason: str = ""):
    color    = COLORS.get(status, "")
    readings = "  ".join(f"{FEATURES[i]}={vals[i]:.2f}{UNITS[i]}"
                         for i in range(N_FEATURES))
    print(f"\n{color}[{status}]{RESET}  state={state}  MSE={loss:.6f}")
    print(f"  Reason: {reason}")
    print(f"  {readings}")
    if trend and status == "NORMAL":
        print(f"  {COLORS['PREDICT']}⚑ PREDICTION: rising error "
              f"(slope={slope:.2e}) — fault may be developing{RESET}")
    if faults:
        print("  Fault details:")
        for f in faults:
            print(f"    → {f['feature']:12s} [{f['type']}]  value={f['value']}")


# ─────────────── DETECTION STATE ──────────
class DetectionState:
    def __init__(self):
        self.current_state  = "unknown"
        self.pending_state  = None
        self.pending_ctr    = 0
        self.buffer         = deque(maxlen=WINDOW_SIZE)
        self.mse_history    = deque(maxlen=HISTORY_SIZE)
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

    def switch_state(self, new_state: str, models: dict):
        self.current_state = new_state
        self.active        = models[new_state]
        self.buffer.clear()
        self.mse_history.clear()
        self.reset_counters()
        print(f"\n  ── State → {new_state.upper()} ──")
        print(f"     fault threshold   : {self.active['threshold']:.6f}")
        print(f"     warning threshold : {self.active['warning_threshold']:.6f}")


# ─────────────── LOAD MODELS ──────────────
print("Loading models...")
state_models: dict = {}
for _s in STATE_BANDS:
    try:
        state_models[_s] = load_state_model(_s)
        cfg = state_models[_s]
        print(f"  ✔ {_s:10s}  fault={cfg['threshold']:.6f}  "
              f"warning={cfg['warning_threshold']:.6f}")
        print(f"             means : {np.round(cfg['feature_means'], 3)}")
        print(f"             stds  : {np.round(cfg['feature_stds'],  3)}")
    except FileNotFoundError as e:
        print(f"  ✘ {e}")

if not state_models:
    sys.exit("No models loaded. Run train_model.py first.")


# ─────────────── SERIAL ───────────────────
print(f"\nConnecting to {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    ser.flushInput()
    print(f"✔ Connected. Starting detection...\n{'─'*60}")
except serial.SerialException as e:
    sys.exit(f"✘ Could not open {PORT}: {e}")


# ─────────────── MAIN LOOP ────────────────
ds = DetectionState()

try:
    while True:
        # ── Read line ──
        try:
            raw = ser.readline().decode(errors="ignore").strip()
        except serial.SerialException:
            print("Serial disconnected — reconnecting in 3 s...")
            time.sleep(3)
            try:
                ser = serial.Serial(PORT, BAUD, timeout=1)
                ser.flushInput()
                print("Reconnected.")
            except serial.SerialException:
                pass
            continue

        if not raw or raw.startswith("#"):
            continue

        parts = raw.split(",")
        if len(parts) != N_FEATURES:
            continue

        try:
            vals = np.array([float(v) for v in parts])
        except ValueError:
            continue

        # ── Hard sanity: reject corrupt voltage ──
        if not (180.0 <= vals[0] <= 260.0):
            continue

        current_reading = vals[1]

        # ── State detection with deadband + confirmation ──
        detected = detect_state(current_reading)

        if detected is not None and detected != ds.current_state:
            if detected == ds.pending_state:
                ds.pending_ctr += 1
            else:
                ds.pending_state = detected
                ds.pending_ctr   = 1

            if ds.pending_ctr >= STATE_CONFIRM_COUNT:
                if detected in state_models:
                    #  CRITICAL FIX: clear previous state data
                    ds.buffer.clear()
                    ds.mse_history.clear()
                    ds.reset_counters()

                    ds.switch_state(detected, state_models)

                ds.pending_state = None
                ds.pending_ctr   = 0

        elif detected == ds.current_state:
            ds.pending_state = None
            ds.pending_ctr   = 0

        if ds.active is None:
            print(f"  Waiting for known state... (current={current_reading:.3f}A)",
                  end="\r")
            continue

        # ── Glitch filter (only after buffer stabilizes) ──
        if len(ds.buffer) >= WINDOW_SIZE:
            if is_glitch(vals, ds.active["feature_means"], ds.active["feature_stds"]):
                print(f"  Glitch rejected: {vals}", end="\r")
                continue

        # ── Safety limit check ──
        sustained_safety = check_safety(vals, ds.safety_ctrs)

        # ── Transition detection (large current jump resets buffer) ──
        if len(ds.buffer) > 0:
            prev_raw = ds.active["scaler"].inverse_transform(
                np.array(ds.buffer[-1]).reshape(1, -1)
            )[0]
            if abs(current_reading - prev_raw[1]) > TRANSITION_DELTA:
                ds.buffer.clear()
                ds.mse_history.clear()
                ds.reset_counters()
                print("↺ Transition detected → buffer reset")
                continue

        # ── Scale and buffer ──
        vals_scaled = np.clip(ds.active["scaler"].transform([vals])[0], 0.0, 1.0)
        ds.buffer.append(vals_scaled)

        if len(ds.buffer) < WINDOW_SIZE:
            print(f"  Buffering... {len(ds.buffer)}/{WINDOW_SIZE}", end="\r")
            continue

        # ── Inference ──
        x = torch.tensor(np.array(ds.buffer).reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            recon = ds.active["model"](x)
            loss  = torch.mean((recon - x) ** 2).item()

        ds.mse_history.append(loss)
        if len(ds.mse_history) < 10:
            reason = "warming up (collecting baseline data)"
            print_status("NORMAL", loss, vals, ds.current_state, reason=reason)
            continue

        is_rising, slope = detect_rising_trend(ds.mse_history)
        thr  = ds.active["threshold"]
        warn = ds.active["warning_threshold"]

        # ─────────────── STATUS MACHINE ───────────────────────────────────────
        # FAULT:   entered after CONFIRMATION_COUNT consecutive MSE > thr
        #          exits only when MSE drops BELOW warning_threshold (conservative)
        # WARNING: entered after WARNING_COUNT consecutive MSE > warn
        #          exits only when MSE drops BELOW warning_threshold
        # NORMAL:  default
        # ──────────────────────────────────────────────────────────────────────

        # ── Feature bandwidth check (NEW) ──
        within_band = is_within_band(
            vals,
            ds.active["feature_means"],
            ds.active["feature_stds"],
            k=3.5
        )

        # ── Updated triggers ──
        if loss > thr or sustained_safety:
            ds.anomaly_ctr += 1
        else:
            ds.anomaly_ctr = 0

        warning_trigger = (
            ((loss > warn*1.5) and not within_band)
            or bool(sustained_safety)
        )


        if ds.anomaly_ctr >= CONFIRMATION_COUNT:
            ds.fault_active   = True
            ds.warning_active = True

        # Persistent FAULT
        if ds.fault_active:
            faults = classify_faults(
                vals, ds.active["feature_means"], ds.active["feature_stds"]
            )
            faults.extend(sustained_safety)
            reason = get_reason(loss, warn, thr, within_band, sustained_safety)
            print_status("FAULT", loss, vals, ds.current_state, faults=faults, reason=reason)

            if loss < thr * 0.9 and within_band and not sustained_safety:
                ds.fault_active   = False
                ds.warning_active = False
                ds.anomaly_ctr    = 0
                ds.warning_ctr    = 0
                ds.mse_history.clear()
            continue

        # Persistent WARNING
        if warning_trigger:
            if loss > warn * 1.2 and not within_band:
                ds.warning_ctr += 1
            else:
                ds.warning_ctr = 0

        if ds.warning_ctr >= WARNING_COUNT:
            ds.warning_active = True

        if ds.warning_active:
            warnings = []
            if loss > warn:
                warnings.append({
                    "feature": "System",
                    "type": "PATTERN_DEVIATION",
                    "value": round(loss, 6)
                })
            warnings.extend(sustained_safety)

            reason = get_reason(loss, warn, thr, within_band, sustained_safety)
            print_status("WARNING", loss, vals, ds.current_state,
                        faults=warnings, reason=reason)

            #  FIX: proper exit condition
            if loss < warn * 0.9 and within_band and not sustained_safety:
                ds.warning_active = False
                ds.warning_ctr = 0

            continue

        # NORMAL (with optional trend prediction)
        reason = get_reason(loss, warn, thr, within_band, sustained_safety)

        if is_rising and loss > warn * 0.7:
            print_status("PREDICT", loss, vals, ds.current_state,
                         trend=True, slope=slope, reason=reason)
        else:
            print_status("NORMAL", loss, vals, ds.current_state,
                         trend=is_rising, slope=slope, reason=reason)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
    ser.close()