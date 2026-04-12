# MotorGuard — Motor Fault Prediction System
Final Year Project · Deep Learning Autoencoder · ESP32 · Real-time Dashboard

---

## Project Structure

```
MOTOR_PM/
├── esp32/
│   └── motor_sensor.ino          ← Flash this to ESP32
├── python/
│   ├── serial_logger.py          ← Step 1: Record data
│   ├── clean_data.py             ← Step 2: Clean & label
│   ├── train_model.py            ← Step 3: Train model
│   ├── detect_realtime.py        ← Step 4a: Terminal detection
│   └── api_server.py             ← Step 4b: Dashboard backend
├── frontend/
│   └── index.html                ← Dashboard (open in browser)
├── data/
│   ├── raw/                      ← Session CSVs go here
│   └── processed/                ← Clean data + plots go here
└── requirements.txt
```

---

## Wiring

| Sensor      | Sensor Pin | ESP32 GPIO | Notes                              |
|-------------|------------|------------|------------------------------------|
| ACS712 30A  | OUT        | GPIO 34    | ADC1 only                          |
|             | VCC        | 3.3V       |                                    |
|             | GND        | GND        |                                    |
| ZMPT101B    | OUT        | GPIO 35    | ADC1 only                          |
|             | VCC        | 3.3V       |                                    |
|             | GND        | GND        |                                    |
| SW-420      | A0 (analog)| GPIO 32    | Use A0 pin, not D0                 |
|             | VCC        | 3.3V       |                                    |
|             | GND        | GND        |                                    |
| DHT11       | DATA       | GPIO 4     | 10kΩ pull-up between DATA and 3.3V |
|             | VCC        | 3.3V       |                                    |
|             | GND        | GND        |                                    |

---

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Calibrate ESP32 (IMPORTANT — do once before recording)

**ACS712 zero-current midpoint:**
1. Upload `motor_sensor.ino` to ESP32
2. Open Arduino Serial Monitor (115200 baud)
3. Disconnect fan — no current flowing
4. Note the raw current ADC value (comment added to sketch)
5. Update `ACS_MIDPOINT` in `motor_sensor.ino` to that value
6. Re-upload

**ZMPT101B voltage scale:**
1. Connect fan's AC supply
2. Measure actual voltage with a multimeter
3. Adjust `VOLT_SCALE` in `motor_sensor.ino` until serial matches multimeter
4. Re-upload

---

## Recording Sessions

```bash
cd python
python serial_logger.py
```

**Recording protocol (repeat 2–3 hours):**

| Step | Action                           | Duration  |
|------|----------------------------------|-----------|
| 1    | Start logger (already running)   | —         |
| 2    | Turn fan ON at Speed 1           | 10 mins   |
| 3    | Increase to Speed 2              | 10 mins   |
| 4    | Increase to Speed 3              | 10 mins   |
| 5    | Turn fan OFF                     | 2–3 mins  |
| 6    | Repeat steps 2–5                 | —         |

Press **s + Enter** for live stats.  
Press **q + Enter** to stop and save.

Each session saves to `data/raw/session_YYYYMMDD_HHMMSS.csv`.

---

## Clean Data

```bash
python clean_data.py
```

After running, check `data/processed/plots/current_distribution.png`.  
You should see 4 clear current clusters (off / speed1 / speed2 / speed3).  
If not, update `SPEED_BANDS` in `clean_data.py` and re-run.

---

## Train Model

```bash
python train_model.py
```

Check `python/training_report/loss_curves.png` — train and val loss should  
converge and flatten (not diverge). This confirms a well-trained model.

---

## Run Dashboard

```bash
# Terminal 1 — start API server
cd python
uvicorn api_server:app --port 8000

# Browser — open dashboard
open frontend/index.html
# (or just double-click index.html in file explorer)
```

**Note:** If browser blocks the API call due to CORS, serve the frontend:
```bash
cd frontend
python -m http.server 3000
# then open http://localhost:3000
```

---

## Run Terminal Detector (without dashboard)

```bash
cd python
python detect_realtime.py
```

---

## Output Files (after training)

| File                      | Contents                          |
|---------------------------|-----------------------------------|
| `python/model.pth`        | Trained autoencoder weights       |
| `python/scaler.pkl`       | MinMaxScaler fitted on train data |
| `python/threshold.npy`    | Fault threshold (95th percentile) |
| `python/warning_threshold.npy` | Warning threshold (85th pct) |
| `python/feature_thresholds.npy` | Per-feature fault thresholds |
| `python/training_report/` | Loss curves, error distribution   |
| `data/processed/plots/`   | Data diagnostics                  |