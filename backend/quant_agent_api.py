# quant_agent_api.py
# -*- coding: utf-8 -*-
"""
Group Project - Group#2
Project: Bicycle Theft Prediction - Flask API

This script:
- Loads the serialized best model, scaler, and feature list using pickle
- Exposes a Flask API with:
    GET  /         -> health check
    GET  /config   -> public runtime config for frontend
    POST /predict  -> returns prediction + probability
"""

import os
import pickle
from pathlib import Path
import json

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ------------------------ LOAD SERIALIZED OBJECTS ------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "bike_theft_best_model.pkl"
SCALER_PATH = BASE_DIR / "artifacts" / "bike_theft_scaler.pkl"
FEATURES_PATH = BASE_DIR / "artifacts" / "bike_theft_selected_features.npy"
METADATA_PATH = BASE_DIR / "artifacts" / "bike_theft_model_metadata.json"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

feature_names = np.load(FEATURES_PATH, allow_pickle=True).tolist()

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
else:
    model_metadata = {}

# ------------------------ FLASK APP SETUP ------------------------

app = Flask(__name__)
CORS(app)  # allow calls from frontend


def preprocess_input_json(input_json: dict):
    """
    Convert raw JSON into a scaled feature vector in the same
    format used to train the model.
    """
    row = {name: input_json.get(name, 0) for name in feature_names}
    x_input = pd.DataFrame([row], columns=feature_names)
    x_scaled = scaler.transform(x_input)
    return x_scaled


@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "message": "Bicycle Theft Prediction API is running.",
            "model_type": type(model).__name__,
            "n_features": len(feature_names),
            "baseline_positive_rate": model_metadata.get("positive_rate_overall"),
        }
    )


@app.route("/config", methods=["GET"])
def public_config():
    """
    Public config consumed by frontend at runtime.
    Note: Google Maps JavaScript keys are client-visible by design,
    so security relies on strict key restrictions in Google Cloud.
    """
    return jsonify(
        {
            "google_maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY", "").strip(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Predict whether a bike is STOLEN (1) or RECOVERED (0)."""
    try:
        data_json = request.get_json()

        if not data_json:
            return jsonify({"error": "Request body must be JSON with feature values."}), 400

        x_new = preprocess_input_json(data_json)

        class_pred = int(model.predict(x_new)[0])

        if hasattr(model, "predict_proba"):
            prob_stolen = float(model.predict_proba(x_new)[0][1])
        else:
            prob_stolen = None

        status_label = "STOLEN" if class_pred == 1 else "RECOVERED"

        return jsonify(
            {
                "prediction_class": class_pred,
                "prediction_label": status_label,
                "probability_stolen": prob_stolen,
                "model_type": type(model).__name__,
                "used_features_order": feature_names,
                "baseline_positive_rate": model_metadata.get("positive_rate_overall"),
                "target_definition": model_metadata.get(
                    "target_definition",
                    "Risk score for STOLEN vs RECOVERED among reported theft cases.",
                ),
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
