"""
check_openmax.py -- sanity checks for the unexpectedly strong OpenMax result (CE outputs only).
  (a) known acceptance on test at tau (should be ~0.95, like every other method)
  (b) where the gain comes from: plain distance to class mean (no Weibull) vs full OpenMax
  (c) sensitivity to tail size and alpha
Usage:
  !python check_openmax.py --runs_dir ".../runs" --seeds 0,1,2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", required=True)
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()


def eucos(v, mav):
    return np.linalg.norm(v - mav, axis=1) / 200.0 + (
        1 - (v @ mav) / (np.linalg.norm(v, axis=1) * np.linalg.norm(mav) + 1e-12))


def fit(tr, tail):
    ms = []
    for c in range(tr["logits"].shape[1]):
        av = tr["logits"][(tr["label"] == c) & (tr["pred"] == c)]
        mav = av.mean(0)
        d = np.sort(eucos(av, mav))[-tail:]
        shape, _, scale = stats.weibull_min.fit(d, floc=0)
        ms.append((mav, shape, scale, d.min(), d.max()))
    return ms


def openmax(lg, ms, alpha):
    n, C = lg.shape
    alpha = min(alpha, C)
    ranks = np.argsort(-lg, 1)
    w = np.ones_like(lg)
    for r in range(alpha):
        for c in np.unique(ranks[:, r]):
            idx = np.where(ranks[:, r] == c)[0]
            mav, sh, sc = ms[c][:3]
            w[idx, c] = 1 - (alpha - r) / alpha * stats.weibull_min.cdf(eucos(lg[idx], mav), sh, loc=0, scale=sc)
    unk = (lg * (1 - w)).sum(1, keepdims=True)
    return 1 - softmax(np.concatenate([unk, lg * w], 1), 1)[:, 0]


def dist_score(lg, ms):  # minus eucos distance to the MAV of the predicted class
    pred = lg.argmax(1)
    d = np.empty(len(lg))
    for c in np.unique(pred):
        idx = pred == c
        d[idx] = eucos(lg[idx], ms[c][0])
    return -d


def auroc(sk, su):
    return roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], -np.r_[sk, su])


rows = []
for s in [int(x) for x in args.seeds.split(",")]:
    d = Path(args.runs_dir) / "ce" / f"seed{s}"
    o = {k: dict(np.load(d / f"outputs_{k}.npz", allow_pickle=True)) for k in ["train", "val", "test_known", "test_unknown"]}
    tr_acc = float((o["train"]["pred"] == o["train"]["label"]).mean())
    for tail in [10, 20, 50, 100]:
        ms = fit(o["train"], tail)
        for alpha in [1, 3, 5, 10, 13]:
            sv, sk, su = (openmax(o[k]["logits"], ms, alpha) for k in ["val", "test_known", "test_unknown"])
            tau = np.quantile(sv, 0.05)
            rows.append({"seed": s, "score": "OpenMax", "tail": tail, "alpha": alpha,
                         "auroc": auroc(sk, su), "known_accept_test": float((sk >= tau).mean()),
                         "unknown_recall": float((su < tau).mean())})
    ms = fit(o["train"], 20)
    sv, sk, su = (dist_score(o[k]["logits"], ms) for k in ["val", "test_known", "test_unknown"])
    tau = np.quantile(sv, 0.05)
    rows.append({"seed": s, "score": "distance_to_MAV_only", "tail": None, "alpha": None,
                 "auroc": auroc(sk, su), "known_accept_test": float((sk >= tau).mean()),
                 "unknown_recall": float((su < tau).mean())})
    lg = lambda k: o[k]["logits"].max(1)
    print(f"seed{s}: train acc {tr_acc:.4f} | MLS auroc {auroc(lg('test_known'), lg('test_unknown')):.4f} | "
          f"Weibull shape/scale (class 0, tail 20) {ms[0][1]:.2f}/{ms[0][2]:.4f}, "
          f"tail range {ms[0][3]:.4f}-{ms[0][4]:.4f}")

df = pd.DataFrame(rows)
df.to_csv(Path(args.runs_dir) / "comparison" / "openmax_checks.csv", index=False)
print("\n(b) distance only vs default OpenMax (tail 20, alpha 10):")
print(df[(df.score != "OpenMax") | ((df["tail"] == 20) & (df["alpha"] == 10))]
      .groupby("score")[["auroc", "known_accept_test", "unknown_recall"]].mean().round(4).to_string())
print("\n(c) OpenMax AUROC by tail x alpha (mean over seeds):")
print(df[df.score == "OpenMax"].pivot_table(index="tail", columns="alpha", values="auroc").round(4).to_string())
print("\n(a) known acceptance on test at tau (OpenMax, mean over seeds):")
print(df[df.score == "OpenMax"].pivot_table(index="tail", columns="alpha", values="known_accept_test").round(3).to_string())
