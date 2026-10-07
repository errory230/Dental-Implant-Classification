"""
eval_baselines.py -- score all methods on identical splits and compare with ARPL.

Outputs (<runs_dir>/comparison/): per_seed.csv, summary.csv, wilcoxon.csv, report.txt
Usage:
  !python eval_baselines.py --runs_dir ".../runs" --split_dir ".../splits"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp, softmax

from osr_metrics import evaluate, threshold_at_tpr

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", required=True)
ap.add_argument("--split_dir", required=True)
ap.add_argument("--tpr", type=float, default=0.95)
ap.add_argument("--tail", type=int, default=20, help="OpenMax Weibull tail size")
ap.add_argument("--alpha", type=int, default=10, help="OpenMax number of top classes to revise")
args = ap.parse_args()
runs = Path(args.runs_dir)
out = runs / "comparison"
out.mkdir(exist_ok=True)


def npz(p):
    return dict(np.load(p, allow_pickle=True))


# ---------------- OpenMax ----------------
def eucos(v, mav):
    return np.linalg.norm(v - mav, axis=1) / 200.0 + (
        1 - (v @ mav) / (np.linalg.norm(v, axis=1) * np.linalg.norm(mav) + 1e-12))


def fit_openmax(tr):
    C = tr["logits"].shape[1]
    models = []
    for c in range(C):
        av = tr["logits"][(tr["label"] == c) & (tr["pred"] == c)]
        mav = av.mean(0)
        d = np.sort(eucos(av, mav))[-args.tail:]
        shape, _, scale = stats.weibull_min.fit(d, floc=0)
        models.append((mav, shape, scale))
    return models


def openmax_score(logits, models):
    n, C = logits.shape
    alpha = min(args.alpha, C)
    ranks = np.argsort(-logits, axis=1)
    w = np.ones_like(logits)
    for r in range(alpha):
        cls = ranks[:, r]
        wr = (alpha - r) / alpha
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            mav, shape, scale = models[c]
            cdf = stats.weibull_min.cdf(eucos(logits[idx], mav), shape, loc=0, scale=scale)
            w[idx, c] = 1 - wr * cdf
    revised = logits * w
    unk = (logits * (1 - w)).sum(1, keepdims=True)
    p = softmax(np.concatenate([unk, revised], 1), axis=1)
    return 1 - p[:, 0], revised.argmax(1)


def scores_ce(o, om):
    lg = o["logits"]
    s = {"MSP": softmax(lg, 1).max(1), "MLS": lg.max(1), "Energy": logsumexp(lg, 1), "ODIN": o["odin_score"]}
    preds = {m: o["pred"] for m in s}
    s["OpenMax"], preds["OpenMax"] = openmax_score(lg, om)
    return s, preds


# ---------------- evaluate ----------------
KEYS = ["closed_acc", "closed_macro_f1", "auroc", "aupr_unknown", "fpr_at_95tpr", "oscr",
        "open_known_macro_f1", "unknown_precision", "unknown_recall", "unknown_f1"]
rows = []
seeds = sorted({p.parent.name for p in (runs / "ce").glob("seed*/DONE")} &
               {p.parent.name for p in (runs / "arpl").glob("seed*/DONE")}, key=lambda s: int(s[4:]))
print(f"seeds with both CE and ARPL finished: {seeds}")
for seed in seeds:
    split = pd.read_csv(Path(args.split_dir) / f"{seed}.csv", dtype=str)[["path", "site_type", "unknown_group"]]
    ce = {s: npz(runs / "ce" / seed / f"outputs_{s}.npz") for s in ["train", "val", "test_known", "test_unknown"]}
    ar = {s: npz(runs / "arpl" / seed / f"outputs_{s}.npz") for s in ["val", "test_known", "test_unknown"]}
    om = fit_openmax(ce["train"])
    S, P = {}, {}
    for s in ["val", "test_known", "test_unknown"]:
        sc, pr = scores_ce(ce[s], om)
        sc["ARPL"], pr["ARPL"] = ar[s]["score"], ar[s]["pred"]
        S[s], P[s] = sc, pr
    C = ce["train"]["logits"].shape[1]
    meta = pd.DataFrame({"path": ce["test_unknown"]["path"]}).merge(split, on="path", how="left")
    shared = (meta.site_type == "shared_site").to_numpy()
    for m in S["val"]:
        # ARPL arrays are in the same split order but check path alignment explicitly
        src = ar if m == "ARPL" else ce
        assert (src["test_unknown"]["path"] == ce["test_unknown"]["path"]).all()
        tau = threshold_at_tpr(S["val"][m], args.tpr)
        yk = src["test_known"]["label"]
        sk, pk = S["test_known"][m], P["test_known"][m]
        su, pu = S["test_unknown"][m], P["test_unknown"][m]
        r = {"seed": seed, "method": m}
        r.update({k: v for k, v in evaluate(sk, pk, yk, su, pu, tau, C).items() if k in KEYS})
        sh = evaluate(sk, pk, yk, su[shared], pu[shared], tau, C)
        r.update({f"shared_{k}": sh[k] for k in ["auroc", "oscr", "fpr_at_95tpr", "unknown_recall",
                                                   "unknown_precision"]})
        for g in ["known_manufacturer", "new_manufacturer"]:
            r[f"det_{g}"] = float((su[(meta.unknown_group == g).to_numpy()] < tau).mean())
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(out / "per_seed.csv", index=False)
order = ["MSP", "MLS", "Energy", "ODIN", "OpenMax", "ARPL"]
metrics = [c for c in df.columns if c not in ("seed", "method")]
summ = df.groupby("method")[metrics].agg(lambda s: f"{s.mean():.3f} ± {s.std(ddof=1):.3f}").reindex(order)
summ.to_csv(out / "summary.csv")

# paired comparison: ARPL vs each baseline across seeds
test_rows = []
for m in order[:-1]:
    for k in ["auroc", "oscr", "shared_auroc", "shared_oscr", "unknown_recall", "closed_acc"]:
        a = df[df.method == "ARPL"].set_index("seed")[k]
        b = df[df.method == m].set_index("seed")[k]
        diff = (a - b.reindex(a.index)).to_numpy()
        n = len(diff)
        ci = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        p = stats.wilcoxon(diff).pvalue if n >= 5 and np.any(diff != 0) else np.nan
        test_rows.append({"baseline": m, "metric": k, "mean_diff_ARPL_minus_baseline": round(diff.mean(), 4),
                          "ci95_low": round(diff.mean() - ci, 4), "ci95_high": round(diff.mean() + ci, 4),
                          "wilcoxon_p": p, "ARPL_better_in_n_seeds": int((diff > 0).sum()), "n_seeds": n})
W = pd.DataFrame(test_rows)
W.to_csv(out / "wilcoxon.csv", index=False)

show = ["closed_acc", "auroc", "oscr", "fpr_at_95tpr", "unknown_recall", "unknown_precision",
        "shared_auroc", "shared_oscr", "shared_unknown_recall", "det_known_manufacturer", "det_new_manufacturer"]
rep = (f"seeds: {len(seeds)}\n\n== Summary (mean ± SD) ==\n{summ[show].T.to_string()}\n\n"
       f"== ARPL minus baseline (paired over seeds; Wilcoxon signed-rank) ==\n{W.to_string(index=False)}")
(out / "report.txt").write_text(rep, encoding="utf-8")
print(rep)
