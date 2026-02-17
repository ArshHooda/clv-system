from pathlib import Path

import pandas as pd

from src.clv.reporting import save_run_artifacts


def test_reporting_files(tmp_path: Path):
    df = pd.DataFrame({"CustomerID": [1, 2], "expected_loss": [2.0, 1.0]})
    paths = save_run_artifacts("2020-01-01", {"a": 1}, {"b": 2}, {"c": 3}, 0.5, df, df, tmp_path)
    assert Path(paths["report_json"]).exists()
