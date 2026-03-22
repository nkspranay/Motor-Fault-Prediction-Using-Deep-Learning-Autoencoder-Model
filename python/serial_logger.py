import serial
import csv
import os
from datetime import datetime

PORT = "COM5"
BAUD = 115200
CSV_PATH = "../data/motor_data.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)

file_exists = os.path.isfile(CSV_PATH)

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["time", "V", "I", "P", "T", "VIB"])

    print("📊 Logging started...")

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            values = line.split(",")
            if len(values) != 5:
                continue

            values = [float(v) for v in values]

            writer.writerow([datetime.now(), *values])
            f.flush()
            print(values)

        except ValueError:
            continue
