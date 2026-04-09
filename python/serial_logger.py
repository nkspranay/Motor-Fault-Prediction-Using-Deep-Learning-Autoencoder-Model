"""
serial_logger.py
=================
Logs ESP32 sensor data to CSV with automatic session management.

Usage:
    python serial_logger.py

Controls (type in terminal while running):
    q + Enter   → Stop logging and save
    s + Enter   → Print current session stats

Output:
    data/session_YYYYMMDD_HHMMSS.csv
"""

import serial
import serial.tools.list_ports
import csv
import os
import sys
import time
import threading
from datetime import datetime

# ─────────────── CONFIG ───────────────────
PORT        = "COM5"        # Change to your port (Linux: /dev/ttyUSB0)
BAUD        = 115200
DATA_DIR    = "../data"
HEADER      = ["timestamp", "voltage", "current", "power",
               "temperature", "humidity", "vibration"]

# ─────────────── AUTO PORT DETECT ─────────
def find_esp32_port():
    """Try to auto-detect ESP32 COM port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        # ESP32 usually shows up as CP210x or CH340
        if any(x in p.description for x in ["CP210", "CH340", "UART", "USB Serial"]):
            print(f"  Auto-detected ESP32 on: {p.device}")
            return p.device
    return None

# ─────────────── STATS TRACKER ────────────
class SessionStats:
    def __init__(self):
        self.start_time  = time.time()
        self.row_count   = 0
        self.error_count = 0
        self.last_values = {}

    def update(self, row: dict):
        self.row_count += 1
        self.last_values = row

    def print_stats(self):
        elapsed = time.time() - self.start_time
        mins    = int(elapsed // 60)
        secs    = int(elapsed % 60)
        rate    = self.row_count / elapsed if elapsed > 0 else 0
        print(f"\n{'─'*45}")
        print(f"  Session time  : {mins:02d}m {secs:02d}s")
        print(f"  Rows logged   : {self.row_count:,}")
        print(f"  Sample rate   : {rate:.1f} rows/sec")
        print(f"  Errors skipped: {self.error_count}")
        if self.last_values:
            v = self.last_values
            print(f"  Last reading  :")
            print(f"    Voltage     : {v.get('voltage','?')} V")
            print(f"    Current     : {v.get('current','?')} A")
            print(f"    Power       : {v.get('power','?')} W")
            print(f"    Temperature : {v.get('temperature','?')} °C")
            print(f"    Humidity    : {v.get('humidity','?')} %")
            print(f"    Vibration   : {v.get('vibration','?')}")
        print(f"{'─'*45}\n")

# ─────────────── KEYBOARD LISTENER ────────
stop_flag   = threading.Event()
stats_flag  = threading.Event()

def keyboard_listener():
    """Background thread to listen for q/s commands."""
    while not stop_flag.is_set():
        try:
            cmd = input().strip().lower()
            if cmd == "q":
                print("\n  Stopping logger...")
                stop_flag.set()
            elif cmd == "s":
                stats_flag.set()
        except EOFError:
            break

# ─────────────── VALIDATE ROW ─────────────
def validate_row(parts: list) -> dict | None:
    """
    Parse and validate a comma-separated sensor line.
    Returns dict on success, None on failure.
    """
    if len(parts) != 6:
        return None
    try:
        voltage     = float(parts[0])
        current     = float(parts[1])
        power       = float(parts[2])
        temperature = float(parts[3])
        humidity    = float(parts[4])
        vibration   = float(parts[5])
    except ValueError:
        return None

    # ── Sanity bounds (REMI fan: 230V AC, max ~2A) ──
    if not (150 <= voltage <= 270):   return None
    if not (0   <= current <= 5):     return None
    if not (0   <= power   <= 1000):  return None
    if not (0   <= temperature <= 80):return None
    if not (0   <= humidity <= 100):  return None
    if not (0   <= vibration <= 4095):return None

    return {
        "voltage"    : round(voltage, 2),
        "current"    : round(current, 4),
        "power"      : round(power, 2),
        "temperature": round(temperature, 1),
        "humidity"   : round(humidity, 1),
        "vibration"  : round(vibration, 1),
    }

# ─────────────── MAIN LOGGER ──────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Port selection ──
    port = find_esp32_port() or PORT
    print(f"\n{'═'*45}")
    print(f"  Motor Fault Detection — Serial Logger")
    print(f"{'═'*45}")
    print(f"  Port    : {port}")
    print(f"  Baud    : {BAUD}")
    print(f"  Output  : {DATA_DIR}/")
    print(f"\n  Commands while running:")
    print(f"    s + Enter  → print stats")
    print(f"    q + Enter  → stop and save")
    print(f"{'═'*45}\n")

    # ── Connect to ESP32 ──
    try:
        ser = serial.Serial(port, BAUD, timeout=2)
        time.sleep(2)   # Let ESP32 reset after serial connect
        ser.flushInput()
        print(f"  ✔ Connected to {port}\n")
    except serial.SerialException as e:
        print(f"  ✘ Could not open {port}: {e}")
        print(f"  Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"    {p.device} — {p.description}")
        sys.exit(1)

    # ── Session file ──
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = os.path.join(DATA_DIR, f"session_{timestamp}.csv")
    stats      = SessionStats()

    # ── Start keyboard listener ──
    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    print(f"  Logging to: {csv_path}")
    print(f"  Waiting for data...\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()

        while not stop_flag.is_set():

            # ── Print stats if requested ──
            if stats_flag.is_set():
                stats.print_stats()
                stats_flag.clear()

            try:
                raw = ser.readline().decode(errors="ignore").strip()
            except serial.SerialException:
                print("  ✘ Serial connection lost.")
                break

            if not raw:
                continue

            # Skip comment/header lines from ESP32
            if raw.startswith("#"):
                continue

            parts = raw.split(",")
            row   = validate_row(parts)

            if row is None:
                stats.error_count += 1
                continue

            # ── Write row ──
            row_with_time = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                **row
            }
            writer.writerow(row_with_time)
            f.flush()   # Ensure data is written even if interrupted
            stats.update(row)

            # ── Live console output ──
            print(
                f"  V={row['voltage']:6.1f}V  "
                f"I={row['current']:5.3f}A  "
                f"P={row['power']:6.1f}W  "
                f"T={row['temperature']:4.1f}°C  "
                f"H={row['humidity']:4.1f}%  "
                f"Vib={row['vibration']:6.0f}",
                end="\r"
            )

    # ── Session complete ──
    ser.close()
    stats.print_stats()
    print(f"\n  ✔ Data saved to: {csv_path}")
    print(f"  Total rows: {stats.row_count:,}\n")

if __name__ == "__main__":
    main()