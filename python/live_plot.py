import serial
import matplotlib.pyplot as plt
from collections import deque

ser = serial.Serial("COM5", 115200, timeout=1)

V = deque(maxlen=50)
I = deque(maxlen=50)

plt.ion()
fig, ax = plt.subplots()

line_v, = ax.plot([], [], label="Voltage")
line_i, = ax.plot([], [], label="Current")

ax.set_title("Live Motor Voltage & Current")
ax.set_xlabel("Samples")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)

print("📈 Live plot started...")

while True:
    try:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue

        parts = raw.split(",")
        if len(parts) < 2:
            continue

        v = float(parts[0])
        i = float(parts[1])

        V.append(v)
        I.append(i)

        line_v.set_data(range(len(V)), V)
        line_i.set_data(range(len(I)), I)

        ax.relim()
        ax.autoscale_view()

        plt.pause(0.01)

    except ValueError:
        continue
    except KeyboardInterrupt:
        break

