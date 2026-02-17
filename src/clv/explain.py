import pandas as pd
import shap
from pathlib import Path
import matplotlib.pyplot as plt


def shap_global_local(xgb_model, X_train: pd.DataFrame, X_test: pd.DataFrame, customer_ids_test: pd.Series):
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # Global summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("artifacts/reports/shap_summary.png", dpi=200)
    plt.close()

    # Local example: highest risk customer
    probs = xgb_model.predict_proba(X_test)[:, 1]
    idx = int(probs.argmax())

    local_customer = int(customer_ids_test.iloc[idx])
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx, :],
        X_test.iloc[idx, :],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"artifacts/reports/shap_force_customer_{local_customer}.png", dpi=200)
    plt.close()

    return {
        "shap_summary_path": "artifacts/reports/shap_summary.png",
        "local_customer_id": local_customer,
        "local_force_path": f"artifacts/reports/shap_force_customer_{local_customer}.png",
    }
