from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def seed(path: str = "data/online_retail_sample.csv", n_customers: int = 120, days: int = 260):
    random.seed(42)
    rows = []
    start = datetime(2010, 1, 1)
    invoice = 530000
    customers = list(range(10000, 10000 + n_customers))
    for d in range(days):
        date = start + timedelta(days=d)
        active_n = random.randint(20, min(60, n_customers))
        active = random.sample(customers, k=active_n)
        for cid in active:
            lines = random.randint(1, 4)
            for _ in range(lines):
                qty = random.choices([1, 2, 3, 4, 5, -1], weights=[20, 20, 20, 20, 18, 2], k=1)[0]
                price = round(random.uniform(0.5, 20.0), 2)
                rows.append(
                    {
                        "InvoiceNo": str(invoice),
                        "StockCode": f"SKU{random.randint(100, 999)}",
                        "Description": "Synthetic Item",
                        "Quantity": qty,
                        "InvoiceDate": date.strftime("%d-%m-%Y %H:%M"),
                        "UnitPrice": price,
                        "CustomerID": int(cid),
                        "Country": "United Kingdom",
                    }
                )
            invoice += 1
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    seed()
