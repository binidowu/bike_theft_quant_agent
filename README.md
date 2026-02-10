# Bicycle Theft Quant Agent (Current Baseline)

This project currently provides a **bike theft risk prediction app** (ML model + Flask API + frontend UI).
It is the baseline that will be expanded into a full quantitative tool-calling agent.
Current target semantics: risk score for `STOLEN (not recovered)` vs `RECOVERED` among reported theft cases.

## Current Structure

- `backend/quant_agent_api.py` -> Flask API for inference (`GET /`, `GET /config`, `POST /predict`)
- `scripts/bike_theft_training_pipeline.py` -> EDA, preprocessing, training, evaluation, serialization
- `data/bicycle_thefts_open_data.csv` -> dataset
- `artifacts/` -> serialized model/scaler/selected features
- `frontend/` -> browser UI (`index.html`, `script.js`, `styles.css`)
- `.env.example` -> env variables template

## What It Currently Does

1. Trains two models (Logistic Regression, Random Forest) on bicycle theft data.
2. Compares metrics and saves the best model artifacts.
3. Serves predictions through Flask API.
4. Provides a frontend form that collects inputs and calls `/predict`.
5. Provides public runtime config (`/config`) used by frontend to optionally load Google Places.

## How It Works

### Training pipeline
`scripts/bike_theft_training_pipeline.py`:
- Loads CSV from `data/`
- Performs EDA + statistics + charts
- Handles missing data + encodes categorical variables
- Uses a stable 12-feature inference schema (time + bike + coordinates)
- Splits data and scales features
- Trains class-weighted models (no SMOTE in final pipeline)
- Trains/evaluates models and saves artifacts to `artifacts/`

### Inference API
`backend/quant_agent_api.py`:
- Loads `artifacts/bike_theft_best_model.pkl`
- Loads scaler and selected feature order
- Exposes `GET /config` for frontend runtime config
- Accepts JSON payload at `POST /predict`
- Reorders/scales input and returns:
  - `prediction_class`
  - `prediction_label`
  - `probability_stolen`
  - `baseline_positive_rate`
  - `target_definition`

### Frontend
`frontend/script.js`:
- Fetches backend config from `GET /config`
- Loads Google Places only if `GOOGLE_MAPS_API_KEY` is provided
- Supports manual latitude/longitude fallback
- Builds 12-feature payload and calls `POST /predict`

## Run Instructions

1. Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. (Optional) Configure `.env`
```bash
cp .env.example .env
# then edit .env and set GOOGLE_MAPS_API_KEY=...
```

4. Regenerate artifacts (recommended)
```bash
MPLBACKEND=Agg python3 scripts/bike_theft_training_pipeline.py
```

5. Start backend
```bash
make run-api
```

6. Serve frontend
```bash
make run-ui
```

7. Open
- `http://127.0.0.1:5500`

## API Quick Test

```bash
curl -X POST "http://127.0.0.1:5005/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "OCC_YEAR": 2024,
    "OCC_DAY": 15,
    "OCC_DOY": 120,
    "OCC_HOUR": 14,
    "REPORT_YEAR": 2024,
    "REPORT_DAY": 16,
    "REPORT_DOY": 121,
    "REPORT_HOUR": 9,
    "BIKE_SPEED": 18,
    "BIKE_COST": 800,
    "LONG_WGS84": -79.38,
    "LAT_WGS84": 43.65
  }'
```

## Known Issues

- Serialized model compatibility depends on `scikit-learn` version.
- If API fails while loading pickle, retrain artifacts with your current environment:
```bash
MPLBACKEND=Agg python3 scripts/bike_theft_training_pipeline.py
```
- `matplotlib` and `imbalanced-learn` are required for training pipeline.

## Notes on Google Maps Key

- In browser-based autocomplete, the key is always visible client-side at runtime.
- The correct security control is **key restriction** in Google Cloud:
  - Restrict APIs to Maps JavaScript API (and only required APIs)
  - Restrict HTTP referrers to your allowed origins
- This project avoids committing the key by sourcing it from backend env (`.env`) and serving it via `/config`.
