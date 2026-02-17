import pandas as pd

from src.clv.decisioning import optimize_targeting


def test_decisioning_constraints():
    df = pd.DataFrame({"CustomerID": range(100), "expected_loss": [100 - i for i in range(100)], "expected_clv": [i for i in range(100)]})
    top, summary = optimize_targeting(df, 10, 1, 5, 0.1, "expected_loss")
    assert len(top) <= 5
    assert summary["total_cost"] <= 10
