# CLV Retention Decision Engine

Production-grade Customer Lifetime Value (CLV) retention system with rolling ML datasets, churn + revenue modeling, versioned prediction store, optimization-based decisioning, standardized reporting, API, and dashboard.

## Architecture (text diagram)
- **Ingestion** (`src/clv/ingest.py`): CSV -> DuckDB `fact_transactions`
- **Feature/label generation** (`src/clv/features_sql.py`, `src/clv/labels.py`) on time windows
- **Rolling builder** (`src/clv/rolling.py`): multi-cutoff `customer_model_data_rollup`
- **Training** (`train_churn.py`, `train_revenue.py`): time-split models + artifacts
- **Scoring** (`score.py`): snapshot tables `predictions_customer_YYYY_MM_DD` + `predictions_customer_latest`
- **Decisioning** (`decisioning.py`): budget/capacity constrained targeting + blending
- **Reporting/Quality** (`reporting.py`, `quality.py`): CSV/JSON artifacts + checks
- **Serving**: FastAPI (`src/api`) and Streamlit (`src/app`)

## Local run
```bash
pip install -r requirements.txt
python scripts/seed_demo_data.py
python -m src.clv.run_all
uvicorn src.api.main:app --reload
streamlit run src/app/app.py
```

## Tests
```bash
pytest -q
```

## Extending
- Add features in `features_sql.py` and include in training (auto-picked).
- Change rolling cadence in `configs/params.yaml` (`step_days`).
- Adjust targeting objective weights in API/UI or config defaults.

## Production-grade qualities
- Time-aware rolling splits and training.
- Versioned prediction store with latest view.
- Reproducible artifacts and configuration-driven runs.
- Quality checks (validation, drift, calibration).
- API + dashboard for operationalization.
- Tests, lint/format, and CI workflow.
