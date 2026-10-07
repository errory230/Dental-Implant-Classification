"""Summarize finished seeds as mean +/- SD.
Usage: !python aggregate.py --run_dir ".../runs/arpl" """
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
args = ap.parse_args()
rows = []
for f in sorted(Path(args.run_dir).glob("seed*/metrics.json")):
    m = json.loads(f.read_text())
    r = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    r.update({f"det_{k}": v["detection_rate"] for k, v in m["detection_by_site_type"].items()})
    r.update({f"det_{k}": v["detection_rate"] for k, v in m["detection_by_unknown_group"].items()})
    r["seed"] = f.parent.name
    rows.append(r)
df = pd.DataFrame(rows).set_index("seed")
summ = pd.DataFrame({"mean": df.mean(), "sd": df.std(ddof=1), "n_seeds": df.count()}).round(4)
print(df.round(4).T.to_string()); print(); print(summ.to_string())
df.to_csv(Path(args.run_dir) / "per_seed.csv"); summ.to_csv(Path(args.run_dir) / "summary.csv")
