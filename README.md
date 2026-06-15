# Aqua Sentinel

A real-time IoT water quality monitoring system that predicts whether your household water is safe — before it reaches your RO unit — using a hybrid CNN-LSTM model trained on 118,000+ sensor readings.

Presented as a conference paper at TQCEBT'26, CHRIST University, Pune Lavasa.

---

## The problem

Most household RO systems filter water. They do not analyse what enters them.

Water stored in terrace tanks is exposed to sunlight. Temperature rises. Contamination occurs upstream. The RO unit has no way to know — it processes whatever arrives.

By the time a membrane degrades or contamination passes through, the damage is done.

Aqua Sentinel is installed at the RO inlet. It monitors water continuously, predicts safety status in real time, and flags unsafe conditions before they reach the filter.

---

## What it does

Sensors read pH, TDS, and temperature every few seconds. Readings flow through a serial pipeline into a Flask backend, where a CNN-LSTM model analyses the last 20 timesteps as a sequence — not as isolated snapshots — and classifies the water:

```
SAFE       probability > 0.70
MODERATE   probability 0.30 – 0.70
UNSAFE     probability < 0.30
```

Predictions are smoothed over a 5-point moving average to suppress transient sensor noise. A Next.js dashboard displays the live safety gauge, trend charts, and an alert log for every UNSAFE event.

The system requires 20 consecutive readings before the first prediction — the sliding window needs a full sequence before inference begins.

---

## Why CNN-LSTM

A single sensor reading tells you the current value. Twenty readings in sequence tell you the direction, rate of change, and pattern — contamination tends to drift, not spike.

CNN layers extract spatial relationships between pH, TDS, and temperature at each timestep. The LSTM layer learns temporal dependencies across the sequence. Together they catch contamination trends that a standard classifier would miss.

---

## Architecture

```
pH Sensor ──┐
TDS Sensor ──┼──► Arduino UNO ──► Serial JSON
Temp Sensor ─┘
                        │
                        ▼
               serial_reader.py
               Reads stream · Parses JSON · Forwards to API
                        │
                        ▼
               Flask Backend (port 8000)
               Rolling buffer (deque 20)
               MinMaxScaler · SimpleImputer
               CNN-LSTM inference
               Moving average smoothing
               REST API: /predict  /latest  /reset
                        │
                        ▼
               Next.js Dashboard
               Live gauge · Trend charts · Alert log
               Polls backend every 5 seconds
               No simulated data
```

Single inference path. The dashboard never generates its own predictions — everything comes from the backend.

---

## Model performance

| Metric | Result |
|---|---|
| Test accuracy | 82.1% |
| Recall — UNSAFE class | 98% |
| Weighted F1 score | 0.83 |
| Inference latency | < 5 ms |

98% recall on the UNSAFE class was the primary optimisation target. A missed unsafe reading is worse than a false alarm.

---

## Dataset

Training used a public IoT aquaponic fish pond dataset (CC BY 4.0) — 118,286 rows of real sensor readings collected over three months. The same three parameters are measured: pH, TDS, water temperature.

Labels were generated using rule-based thresholds aligned with household water quality standards:

```
SAFE:   6.5 ≤ pH ≤ 8.5  ·  TDS ≤ 500 ppm  ·  24°C ≤ temp ≤ 27°C
UNSAFE: any reading outside the above thresholds
```

The dataset was converted into sliding windows of 20 timesteps, producing an input shape of `(505710, 20, 3)`.

The aquaponic origin is a known limitation — the model is validated on fish pond dynamics used as a proxy for household water variability. Domain adaptation for household-specific conditions is on the roadmap.

---

## Stack

| Layer | Technology |
|---|---|
| ML | TensorFlow · Keras · Scikit-learn |
| Backend | Python 3.10+ · Flask |
| Serial | pyserial |
| Frontend | Next.js · React · TypeScript · Tailwind CSS · Recharts |
| Hardware | Arduino UNO · pH sensor · TDS sensor · DS18B20 |

---

## Running locally

Prerequisites: Python 3.10+, Node.js 20+, Arduino IDE, a connected Arduino with sensors.

**1. Clone and install backend dependencies**
```bash
git clone https://github.com/MadhanShankarG/aqua-sentinel.git
cd aqua-sentinel
pip install flask tensorflow scikit-learn numpy pandas pyserial
```

**2. Start the Flask backend**
```bash
cd backend
python app.py
```

**3. Connect Arduino and start the serial reader**

Flash `arduino/aqua_sentinel.ino` to your Arduino via Arduino IDE, then:
```bash
python backend/serial_reader.py
```

The serial reader connects to Arduino, reads the JSON stream, and forwards each reading to `/predict`.

**4. Install and run the frontend**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard begins polling automatically. The safety gauge activates after the first 20 readings are buffered.

---

## API

**`POST /predict`**

Accepts a sensor reading, appends to the rolling buffer, and returns a prediction once the buffer reaches 20.

```json
// Request
{ "water_pH": 7.21, "TDS": 245.8, "water_temp": 26.4 }

// Response
{ "probability": 0.84, "status": "SAFE", "history": [...] }
```

**`GET /latest`** — returns the most recent prediction from live Arduino data.

**`POST /reset`** — clears buffer, history, and smoothing queue.

**`GET /`** — health check.

---

## Project structure

```
aqua-sentinel/
├── backend/
│   ├── app.py                  # Flask API and inference engine
│   └── serial_reader.py        # Arduino serial interface
├── model/
│   ├── pond_cnn_lstm_model.keras
│   ├── pond_scaler.pkl
│   └── pond_imputer.pkl
├── frontend/
│   └── src/components/
│       ├── dashboard.tsx
│       ├── status-card.tsx
│       ├── stats-cards.tsx
│       ├── charts-panel.tsx
│       ├── alerts-panel.tsx
│       └── sensor-form.tsx
├── arduino/
│   └── aqua_sentinel.ino
└── notebooks/
    └── training.ipynb
```

---

## Research

> **TQCEBT'26** — International Conference on Technology, Quality, Computing, Electronics, Business and Technology  
> CHRIST University, Pune Lavasa Campus

---

## License

MIT. Dataset licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

