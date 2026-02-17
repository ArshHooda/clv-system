## Run end-to-end
1) Configure rolling cutoffs in configs/params.yaml (rolling.enabled: true)
2) Build dataset, train models, score predictions:
   python src/clv/run_all.py

## Outputs
- DuckDB:
  - customer_model_data_rollup
  - predictions_customer (prediction store)
- Artifacts:
  - artifacts/models/churn_xgb.joblib
  - artifacts/models/spend_clf.joblib
  - artifacts/models/revenue_reg.joblib
  - artifacts/models/feature_cols.joblib
- Decisioning:
  python src/clv/tmp_decisioning_report.py
  python src/clv/tmp_weight_sweep.py
