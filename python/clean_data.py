"""
clean_data.py
==============
Merges all session CSVs from data/raw/, cleans them,
auto-labels speed states from current thresholds, and
outputs a single training-ready CSV to data/processed/.

Usage:
    python clean_data.py

Output:
    data/processed/motor_data_clean.csv
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "motor_data_clean.csv")
PLOTS_DIR     = os.path.join(PROCESSED_DIR, "plots")

# ─────────────────────────────────────────────────────────────
# SPEED THRESHOLDS  ← Update these after your first session
# ─────────────────────────────────────────────────────────────
# How to tune: after first session, run the script once and
# check data/processed/current_distribution.png — the histogram
# will show clear clusters. Set the boundaries between clusters.
#
# REMI fan 230V estimated draws:
#   Off     : 0.00 – 0.05 A
#   Speed 1 : 0.05 – 0.30 A  (slowest)
#   Speed 2 : 0.30 – 0.70 A  (medium)
#   Speed 3 : 0.70 – 2.00 A  (fastest)
#
SPEED_THRESHOLDS = {
    "off":     (0.00, 0.05),
    "speed_1": (0.05, 0.30),
    "speed_2": (0.30, 0.70),
    "speed_3": (0.70, 2.00),
}
# ─────────────────────────────────────────────────────────────

# Paths
#RAW_DIR       = "../data/raw"
#PROCESSED_DIR = "../data/processed"
#OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "motor_data_clean.csv")
#PLOTS_DIR     = os.path.join(PROCESSED_DIR, "plots")

# Physical sanity bounds — rows outside these are sensor errors
BOUNDS = {
    "voltage"    : (150.0, 270.0),
    "current"    : (0.0,   5.0),
    "power"      : (0.0,   1200.0),
    "temperature": (10.0,  80.0),
    "humidity"   : (0.0,   100.0),
    "vibration"  : (0.0,   4095.0),
}

FEATURES = ["voltage", "current", "power", "temperature", "humidity", "vibration"]


# ─────────────── HELPERS ──────────────────

def label_speed(current: float) -> str:
    """Assign speed label based on current draw."""
    for label, (lo, hi) in SPEED_THRESHOLDS.items():
        if lo <= current < hi:
            return label
    return "unknown"


def print_section(title: str):
    print(f"\n{'═' * 50}")
    print(f"  {title}")
    print(f"{'═' * 50}")


# ─────────────── LOAD ─────────────────────

def load_sessions() -> pd.DataFrame:
    print_section("1. Loading session files")

    pattern = os.path.join(RAW_DIR, "session_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No session_*.csv files found in {RAW_DIR}/\n"
            f"Run serial_logger.py first to record data."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df["session"] = os.path.basename(f).replace(".csv", "")
        frames.append(df)
        print(f"  ✔ {os.path.basename(f):40s}  {len(df):>6,} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"\n  Total rows loaded : {len(combined):,}")
    print(f"  Sessions          : {len(files)}")
    return combined


# ─────────────── CLEAN ────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print_section("2. Cleaning")

    original = len(df)

    # ── Drop missing values ──
    df = df.dropna(subset=FEATURES)
    print(f"  After dropna         : {len(df):,}  (dropped {original - len(df):,})")

    # ── Remove duplicate timestamps ──
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"])
    print(f"  After dedup          : {len(df):,}  (dropped {before - len(df):,})")

    # ── Apply sanity bounds ──
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in BOUNDS.items():
        col_mask = df[col].between(lo, hi)
        bad = (~col_mask).sum()
        if bad > 0:
            print(f"  Out-of-range {col:12s}: {bad:,} rows removed")
        mask &= col_mask
    df = df[mask]
    print(f"  After bounds filter  : {len(df):,}  (dropped {before - len(df):,})")

    # ── Remove electrical zeros while motor should be running ──
    # Short spikes of 0 voltage/current during running are sensor noise
    before = len(df)
    df = df[~((df["voltage"] < 10) & (df["current"] > 0.1))]
    print(f"  After spike filter   : {len(df):,}  (dropped {before - len(df):,})")

    # ── Rolling median smoothing on vibration (noisy analog sensor) ──
    df = df.copy()
    df["vibration"] = (
        df["vibration"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
    )

    # ── Recalculate power from cleaned V and I ──
    df["power"] = df["voltage"] * df["current"]

    print(f"\n  Final clean rows     : {len(df):,}")
    print(f"  Total removed        : {original - len(df):,} ({(original - len(df)) / original * 100:.1f}%)")
    return df


# ─────────────── LABEL SPEED ──────────────

def add_speed_labels(df: pd.DataFrame) -> pd.DataFrame:
    print_section("3. Auto-labelling speed states")

    df = df.copy()
    df["speed_state"] = df["current"].apply(label_speed)

    counts = df["speed_state"].value_counts()
    for state, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {state:10s} : {count:>7,} rows  ({pct:5.1f}%)")

    unknown = (df["speed_state"] == "unknown").sum()
    if unknown > 0:
        print(f"\n  ⚠ {unknown:,} rows labelled 'unknown' — current outside all thresholds.")
        print(f"    Check SPEED_THRESHOLDS in this script and update after reviewing")
        print(f"    data/processed/plots/current_distribution.png")

    return df


# ─────────────── TRANSITION REMOVAL ───────

def remove_transitions(df: pd.DataFrame, transition_secs: int = 5) -> pd.DataFrame:
    """
    Remove rows within `transition_secs` seconds of a speed state change.
    These transient rows (startup, speed change, shutdown) are ambiguous
    and would confuse the autoencoder during training.
    They are kept in the file with a flag so you can choose to
    use or exclude them during training.
    """
    print_section("4. Flagging transition windows")

    df = df.copy()
    df["is_transition"] = False

    # Find indices where speed_state changes
    state_changed = df["speed_state"] != df["speed_state"].shift(1)
    change_indices = df.index[state_changed].tolist()

    # Mark rows within transition_secs of each change
    flagged = 0
    for idx in change_indices:
        loc = df.index.get_loc(idx)
        # Estimate rows to flag: ~10 rows/sec * transition_secs
        window = transition_secs * 10
        start = max(0, loc - window // 2)
        end   = min(len(df), loc + window)
        df.iloc[start:end, df.columns.get_loc("is_transition")] = True
        flagged += end - start

    print(f"  Transition windows flagged : {flagged:,} rows")
    print(f"  Stable state rows          : {(~df['is_transition']).sum():,} rows")
    print(f"  (Training will use stable rows only; transitions kept for analysis)")

    return df


# ─────────────── PLOTS ────────────────────

def save_plots(df: pd.DataFrame):
    print_section("5. Saving diagnostic plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    colors = {
        "off"    : "#94a3b8",
        "speed_1": "#34d399",
        "speed_2": "#60a5fa",
        "speed_3": "#f87171",
        "unknown": "#fbbf24",
    }

    # ── Current distribution ──
    fig, ax = plt.subplots(figsize=(10, 4))
    for state in df["speed_state"].unique():
        subset = df[df["speed_state"] == state]["current"]
        ax.hist(subset, bins=80, alpha=0.6,
                label=state, color=colors.get(state, "gray"))
    for label, (lo, hi) in SPEED_THRESHOLDS.items():
        ax.axvline(lo, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Current Distribution by Speed State")
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "current_distribution.png"), dpi=150)
    plt.close()
    print(f"  ✔ current_distribution.png")

    # ── Feature overview ──
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        for state in df["speed_state"].unique():
            subset = df[df["speed_state"] == state]
            ax.plot(subset.index, subset[feat],
                    alpha=0.4, linewidth=0.5,
                    color=colors.get(state, "gray"),
                    label=state)
        ax.set_title(feat.capitalize())
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    plt.suptitle("Feature Overview by Speed State", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_overview.png"), dpi=150)
    plt.close()
    print(f"  ✔ feature_overview.png")

    # ── Correlation heatmap ──
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[FEATURES].corr()
    im   = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"  ✔ correlation_heatmap.png")


# ─────────────── SUMMARY ──────────────────

def print_summary(df: pd.DataFrame):
    print_section("6. Dataset Summary")
    print(f"  Shape       : {df.shape}")
    duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60
    print(f"  Duration    : {duration:.1f} minutes")
    print(f"\n  Feature statistics (stable rows only):")
    stable = df[~df["is_transition"]][FEATURES]
    stats  = stable.describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())


# ─────────────── MAIN ─────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_sessions()
    df = clean(df)
    df = add_speed_labels(df)
    df = remove_transitions(df, transition_secs=5)

    save_plots(df)
    print_summary(df)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✔ Clean data saved to: {OUTPUT_FILE}")
    print(f"  Columns: {list(df.columns)}\n")


if __name__ == "__main__":
    main()