from __future__ import annotations
import pandas as pd
from pathlib import Path

def get_data(method: str) -> pd.DataFrame:
    parent_path = Path("../data/results")
    file_name = parent_path / f"{method}_summary.csv"
    with open(file_name, 'r') as f:
        content = f.read()
        runs = content.strip().split(',--- NEW RUN ---')
        records = []
        for run in runs:
            run = run.strip()
            if not run:
                continue
            record = {}
            for line in run.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                key, value = parts[0], parts[1]
                record[key] = value
            records.append(record)
    df = pd.DataFrame(records)
    df = df.apply(pd.to_numeric)
    return df