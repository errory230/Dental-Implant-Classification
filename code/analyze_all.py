"""
analyze_all.py -- aggregate all finished seeds into paper-ready tables (mean +/- SD over seeds).

Inputs : <run_dir>/seed*/ (outputs_*.npz, metrics.json), <split_dir>/seed*.csv, index_final.csv
Outputs: <run_dir>/paper/
  main_metrics.csv          overall + shared-site-only + implant-level metrics
  known_per_class.csv       closed-set precision/recall/F1 per known class (manuscript Table 2 style)
  unknown_per_class.csv     detection rate per unknown class (+ images, implants, report flag)
  unknown_group_site.csv    detection: unknown_group x site_type
  undetected_assignment.csv undetected unknowns -> assigned known class (pooled counts, row %)
  confusion_known_pooled.csv
  report.txt

Usage:
  !python analyze_all.py --run_dir ".../runs/arpl" --split_dir ".../splits" --index ".../index/index_final.csv"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from osr_metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--split_dir", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--min_images", type=int, default=30)
ap.add_argument("--min_implants", type=int, default=10)
args = ap.parse_args()
run, out = Path(args.run_dir), Path(args.run_dir) / "paper"
out.mkdir(exist_ok=True)
index = pd.read_csv(args.index, dtype=str)[["path", "mfr_code"]]
mfr_of = pd.read_csv(args.index, dtype=str).groupby("model_code").mfr_code.first().to_dict()


def ms(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


main, kpc, upc, gs, cms, assign = [], [], [], [], [], []
seed_dirs = sorted(run.glob("seed*/metrics.json"), key=lambda p: int(p.parent.name[4:]))
for mf in seed_dirs:
    d, seed = mf.parent, mf.parent.name
    m = json.loads(mf.read_text())
    tau, classes = m["tau"], m["classes"]
    split = pd.read_csv(Path(args.split_dir) / f"{seed}.csv", dtype=str)

    def load(name):
        z = np.load(d / f"outputs_{name}.npz", allow_pickle=True)
        df = pd.DataFrame({"path": z["path"], "score": z["score"], "pred": z["pred"], "label": z["label"]})
        return df.merge(split, on="path", validate="one_to_one").merge(index, on="path", validate="one_to_one")

    k, u = load("test_known"), load("test_unknown")
    k["pred_cls"] = [classes[i] for i in k.pred]
    u["pred_cls"] = [classes[i] for i in u.pred]
    u["detected"] = u.score < tau
    C = len(classes)

    r = {"seed": seed}
    r.update({key: m[key] for key in ["closed_acc", "closed_macro_f1", "auroc", "aupr_unknown", "fpr_at_95tpr",
                                      "oscr", "open_known_macro_f1", "unknown_precision", "unknown_recall",
                                      "unknown_f1", "known_accept_rate_test"]})
    # manufacturer-level accuracy and share of errors within the same manufacturer
    pred_mfr = k.pred_cls.map(mfr_of)
    r["manufacturer_acc"] = float((pred_mfr == k.mfr_code).mean())
    err = k.pred_cls != k.model_code
    r["errors_within_manufacturer"] = float((pred_mfr[err] == k.mfr_code[err]).mean())
    # shared-site-only (conservative main result)
    us = u[u.site_type == "shared_site"]
    ms_ = evaluate(k.score.values, k.pred.values, k.label.values, us.score.values, us.pred.values, tau, C)
    for key in ["auroc", "aupr_unknown", "fpr_at_95tpr", "oscr", "unknown_precision", "unknown_recall", "unknown_f1"]:
        r[f"shared_{key}"] = ms_[key]
    # implant level
    ik, iu = k.groupby("implant_id").score.mean(), u.groupby("implant_id").score.mean()
    r["implant_auroc"] = float(roc_auc_score(np.r_[np.zeros(len(ik)), np.ones(len(iu))], -np.r_[ik, iu]))
    r["implant_unknown_detection"] = float((iu < tau).mean())
    main.append(r)

    p, rc, f, n = precision_recall_fscore_support(k.model_code, k.pred_cls, labels=classes, zero_division=0)
    kpc.append(pd.DataFrame({"class": classes, "precision": p, "recall": rc, "f1": f, "n": n, "seed": seed}))
    t = u.groupby("model_code").agg(det=("detected", "mean")).reset_index()
    t["seed"] = seed
    upc.append(t)
    g = u.groupby(["unknown_group", "site_type"]).detected.mean().reset_index()
    g["seed"] = seed
    gs.append(g)
    cms.append(confusion_matrix(k.model_code, k.pred_cls, labels=classes))
    assign.append(pd.crosstab(u.loc[~u.detected, "model_code"], u.loc[~u.detected, "pred_cls"]))

n_seeds = len(main)
M = pd.DataFrame(main).set_index("seed")
M.to_csv(out / "main_metrics_per_seed.csv")
summary = pd.DataFrame({"mean": M.mean(), "sd": M.std(ddof=1)}).round(4)
summary["mean ± sd"] = [ms(M[c]) for c in M.columns]
summary.to_csv(out / "main_metrics.csv")

K = pd.concat(kpc)
kt = K.groupby("class").agg(precision=("precision", ms), recall=("recall", ms), f1=("f1", ms),
                            n_test_mean=("n", "mean")).round(1)
kt.to_csv(out / "known_per_class.csv")

# unknown per class with dataset size (images / implants over the whole unknown set)
full = pd.read_csv(Path(args.split_dir) / f"{seed_dirs[0].parent.name}.csv", dtype=str)
full = full[full.role == "unknown"].merge(index, on="path")
size = full.groupby("model_code").agg(images=("path", "size"), implants=("implant_id", "nunique"),
                                      manufacturer=("mfr_code", "first"),
                                      unknown_group=("unknown_group", "first"),
                                      frac_unknown_only_site=("site_type",
                                                              lambda s: round((s == "unknown_only_site").mean(), 2)))
U = pd.concat(upc).groupby("model_code").det.agg(["mean", "std"])
ut = size.join(U)
ut["detection"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(ut["mean"], ut["std"].fillna(0))]
ut["report_individually"] = (ut.images >= args.min_images) & (ut.implants >= args.min_implants)
ut = ut.sort_values("mean", ascending=False)
ut.to_csv(out / "unknown_per_class.csv")

G = pd.concat(gs).groupby(["unknown_group", "site_type"]).detected.agg(ms)
G.to_csv(out / "unknown_group_site.csv")

cm = pd.DataFrame(sum(cms), index=classes, columns=classes)
cm.to_csv(out / "confusion_known_pooled.csv")
A = pd.concat(assign).groupby(level=0).sum().fillna(0)
A.to_csv(out / "undetected_assignment.csv")
Ap = A.div(A.sum(axis=1), axis=0)
rows = []
for c in Ap.index:
    top = Ap.loc[c].sort_values(ascending=False)
    rows.append({"unknown": c, "mfr": mfr_of.get(c, ""), "undetected_total": int(A.loc[c].sum()),
                 "top1": f"{top.index[0]} [{mfr_of.get(top.index[0], '')}] ({top.iloc[0]:.0%})",
                 "top2": f"{top.index[1]} [{mfr_of.get(top.index[1], '')}] ({top.iloc[1]:.0%})"})
top_tab = pd.DataFrame(rows).sort_values("undetected_total", ascending=False)

L = [f"seeds: {n_seeds}", "\n== Main metrics (mean ± SD) ==", summary["mean ± sd"].to_string(),
     "\n== Known classes (closed-set) ==", kt.to_string(),
     "\n== Unknown detection: group x site ==", G.to_string(),
     "\n== Unknown classes ==",
     ut[["manufacturer", "unknown_group", "images", "implants", "frac_unknown_only_site", "detection",
         "report_individually"]].to_string(),
     "\n== Undetected unknowns -> assigned known class (pooled over seeds) ==", top_tab.to_string(index=False)]
rep = "\n".join(L)
(out / "report.txt").write_text(rep, encoding="utf-8")
print(rep)
