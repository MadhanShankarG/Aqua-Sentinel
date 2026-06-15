from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, TypedDict

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

WINDOW_SIZE = 20
HISTORY_SIZE = 20
SMOOTH_WINDOW = 5

BASE_DIR = Path(__file__).resolve().parent


def resolve_artifact(filename: str) -> Path:
    local = BASE_DIR / filename

    if local.exists():
        return local

    parent = BASE_DIR.parent / filename

    if parent.exists():
        return parent

    return local


MODEL_PATH = resolve_artifact("pond_cnn_lstm_model.keras")
SCALER_PATH = resolve_artifact("pond_scaler.pkl")
IMPUTER_PATH = resolve_artifact("pond_imputer.pkl")

print(f"Loading model: {MODEL_PATH}")
print(f"Loading scaler: {SCALER_PATH}")
print(f"Loading imputer: {IMPUTER_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
imputer = joblib.load(IMPUTER_PATH)

buffer: Deque[List[float]] = deque(maxlen=WINDOW_SIZE)
history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_SIZE)
smooth: Deque[float] = deque(maxlen=SMOOTH_WINDOW)


class SensorPayload(TypedDict):
    water_pH: float
    TDS: float
    water_temp: float


def classify(probability: float) -> str:
    if probability < 0.30:
        return "UNSAFE"

    if probability <= 0.70:
        return "MODERATE"

    return "SAFE"


def smooth_probability(probability: float) -> float:
    smooth.append(probability)
    return float(np.mean(smooth))


def create_prediction(probability: float) -> Dict[str, Any]:
    return {
        "probability": probability,
        "status": classify(probability),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_payload(payload: Any) -> SensorPayload:
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON payload")

    try:
        return {
            "water_pH": float(payload["water_pH"]),
            "TDS": float(payload["TDS"]),
            "water_temp": float(payload["water_temp"]),
        }

    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Payload must include numeric water_pH, TDS and water_temp"
        ) from exc


def run_model_on_buffer() -> Dict[str, Any]:
    x = np.asarray(buffer, dtype=np.float32)

    x = imputer.transform(x)
    x = scaler.transform(x)

    x = x.reshape(1, WINDOW_SIZE, 3)

    raw_probability = float(
        model.predict(x, verbose=0)[0][0]
    )

    probability = smooth_probability(raw_probability)

    prediction = create_prediction(probability)

    history.append(prediction)

    return prediction


@app.get("/")
def home():
    return "Backend running"


@app.post("/predict")
def predict():
    try:
        payload = parse_payload(
            request.get_json(silent=True)
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    reading = [
        payload["water_pH"],
        payload["TDS"],
        payload["water_temp"],
    ]

    buffer.append(reading)

    if len(buffer) < WINDOW_SIZE:
        return jsonify(
            {
                "probability": None,
                "status": "COLLECTING",
                "count": len(buffer),
                "required": WINDOW_SIZE,
                "history": list(history),
            }
        )

    prediction = run_model_on_buffer()

    return jsonify(
        {
            "probability": prediction["probability"],
            "status": prediction["status"],
            "history": list(history),
        }
    )


@app.get("/latest")
def latest():
    if not history:
        return jsonify(
            {
                "probability": None,
                "status": "COLLECTING",
                "history": [],
            }
        )

    latest_prediction = history[-1]

    return jsonify(
        {
            "probability": latest_prediction["probability"],
            "status": latest_prediction["status"],
            "history": list(history),
        }
    )


@app.post("/reset")
def reset():
    buffer.clear()
    history.clear()
    smooth.clear()

    return jsonify(
        {
            "message": "Reset complete"
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "window_size": WINDOW_SIZE,
            "history_size": len(history),
            "buffer_size": len(buffer),
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
    )