import pandas as pd
import time
import numpy as np

from detect_realtime import (
    DetectionState,
    state_models,
    is_glitch,
    detect_state,
    check_safety,
    classify_faults,
    detect_rising_trend,
    is_within_band,
    get_reason,
    print_status,
    FEATURES,
    WINDOW_SIZE,
    CONFIRMATION_COUNT,
    WARNING_COUNT
)

# ─────────────── CONFIG ───────────────
CSV_PATH = "faulty_data.csv"
DELAY    = 0.2   # simulate real-time speed


# ─────────────── LOAD DATA ───────────────
df = pd.read_csv(CSV_PATH)

# Ensure correct columns
df = df[["voltage","current","power","temperature","humidity","vibration"]]


# ─────────────── INIT ───────────────
ds = DetectionState()

print("\n📊 Running CSV-based simulation...\n")


# ─────────────── MAIN LOOP ───────────────
for _, row in df.iterrows():

    vals = np.array([
        row["voltage"],
        row["current"],
        row["power"],
        row["temperature"],
        row["humidity"],
        row["vibration"]
    ], dtype=float)

    current_reading = vals[1]

    # ── STATE DETECTION ──
    detected = detect_state(current_reading)

    if detected and detected != ds.current_state:
        if detected in state_models:
            ds.switch_state(detected, state_models)

    if ds.active is None:
        continue

    # ── GLITCH FILTER ──
    if len(ds.buffer) >= WINDOW_SIZE:
        if is_glitch(vals, ds.active["feature_means"], ds.active["feature_stds"]):
            continue

    # ── SAFETY CHECK ──
    sustained_safety = check_safety(vals, ds.safety_ctrs)

    # ── SCALE + BUFFER ──
    vals_scaled = np.clip(ds.active["scaler"].transform([vals])[0], 0.0, 1.0)
    ds.buffer.append(vals_scaled)

    if len(ds.buffer) < WINDOW_SIZE:
        continue

    # ── INFERENCE ──
    import torch
    x = torch.tensor(np.array(ds.buffer).reshape(1, -1), dtype=torch.float32)

    with torch.no_grad():
        recon = ds.active["model"](x)
        loss  = torch.mean((recon - x) ** 2).item()

    ds.mse_history.append(loss)

    if len(ds.mse_history) < 10:
        continue

    is_rising, slope = detect_rising_trend(ds.mse_history)

    thr  = ds.active["threshold"]
    warn = ds.active["warning_threshold"]

    within_band = is_within_band(
        vals,
        ds.active["feature_means"],
        ds.active["feature_stds"],
        k=3.5
    )

    # ── ANOMALY COUNTER ──
    if loss > thr or sustained_safety:
        ds.anomaly_ctr += 1
    else:
        ds.anomaly_ctr = 0

    warning_trigger = (
        ((loss > warn * 1.5) and not within_band)
        or bool(sustained_safety)
    )

    # ── FAULT ──
    if ds.anomaly_ctr >= CONFIRMATION_COUNT:
        ds.fault_active   = True
        ds.warning_active = True

    if ds.fault_active:
        faults = classify_faults(
            vals, ds.active["feature_means"], ds.active["feature_stds"]
        )
        faults.extend(sustained_safety)

        reason = get_reason(loss, warn, thr, within_band, sustained_safety)

        print_status("FAULT", loss, vals, ds.current_state,
                     faults=faults, reason=reason)

        if loss < thr * 0.9 and within_band and not sustained_safety:
            ds.fault_active = False
            ds.warning_active = False
            ds.anomaly_ctr = 0
            ds.warning_ctr = 0
            ds.mse_history.clear()

        time.sleep(DELAY)
        continue

    # ── WARNING ──
    if warning_trigger:
        if loss > warn * 1.2 and not within_band:
            ds.warning_ctr += 1
        else:
            ds.warning_ctr = 0

    if ds.warning_ctr >= WARNING_COUNT:
        ds.warning_active = True

    if ds.warning_active:
        reason = get_reason(loss, warn, thr, within_band, sustained_safety)

        print_status("WARNING", loss, vals, ds.current_state,
                     faults=sustained_safety, reason=reason)

        if loss < warn * 0.9 and within_band and not sustained_safety:
            ds.warning_active = False
            ds.warning_ctr = 0

        time.sleep(DELAY)
        continue

    # ── NORMAL ──
    reason = get_reason(loss, warn, thr, within_band, sustained_safety)

    print_status("NORMAL", loss, vals, ds.current_state, reason=reason)

    time.sleep(DELAY)