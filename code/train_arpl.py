"""
Train ConvNeXt + ARPL for one seed, then save model outputs and sanity-check metrics.
Resumable: checkpoints after every epoch; finished seeds (DONE file) are skipped.

Output per seed (<out_dir>/arpl/seed{k}/):
  best.pt            model + reciprocal points + radius + classes + tau
  log.csv            per-epoch train loss / val accuracy / time
  outputs_{val,test_known,test_unknown}.npz   logits, scores, preds, labels, features(fp16), paths
  metrics.json       sanity-check metrics (+ detection rate by site_type / unknown_group / class)

Usage (Colab):
  !python train_arpl.py --split_csv ".../splits/seed0.csv" --data_root /content/data/panorama \
      --out_dir ".../runs" --seed 0
  Smoke test first:  add --limit 256 --epochs 1 --run_name smoke
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arpl import ARPLoss
from common import atomic_save, build_backbone, load_split, make_loader, seed_everything
from osr_metrics import evaluate, rejection_rate_by, threshold_at_tpr

ap = argparse.ArgumentParser()
ap.add_argument("--split_csv", required=True)
ap.add_argument("--data_root", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--run_name", default="arpl")
ap.add_argument("--backbone", default="convnext_small.fb_in22k")
ap.add_argument("--pretrained", type=int, default=1)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--weight_decay", type=float, default=0.05)
ap.add_argument("--crit_lr", type=float, default=1e-3, help="lr for reciprocal points and radius")
ap.add_argument("--weight_pl", type=float, default=0.1, help="lambda for the open-space loss")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--tpr", type=float, default=0.95)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--amp", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="subsample each split (smoke test)")
args = ap.parse_args()

out = Path(args.out_dir) / args.run_name / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
if (out / "DONE").exists():
    print(f"[skip] {out} already finished")
    raise SystemExit
seed_everything(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
use_amp = bool(args.amp) and dev == "cuda"

# ---------------- data ----------------
parts, classes = load_split(args.split_csv, args.limit, args.seed)
c2i = {c: i for i, c in enumerate(classes)}
print(f"classes ({len(classes)}): {classes}")
print({k: len(v) for k, v in parts.items()})
L = lambda s, train: make_loader(parts[s], args.data_root, c2i, train, args.batch_size, args.workers)
train_loader, val_loader = L("train", True), L("val", False)

# ---------------- model ----------------
model, feat_dim = build_backbone(args.backbone, bool(args.pretrained))
crit = ARPLoss(len(classes), feat_dim, args.weight_pl, args.temp)
model, crit = model.to(dev), crit.to(dev)
opt = torch.optim.AdamW([
    {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
    {"params": crit.parameters(), "lr": args.crit_lr, "weight_decay": 0.0}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def forward(x, y=None):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        f = model(x)
    return f.float(), crit(f.float(), y)  # ARPL distances in fp32


@torch.no_grad()
def infer(loader):
    model.eval(); crit.eval()
    F_, Lg, Y = [], [], []
    for x, y in loader:
        f, (logits, _) = forward(x.to(dev, non_blocking=True))
        F_.append(f.half().cpu()); Lg.append(logits.cpu()); Y.append(y)
    feats, logits, y = torch.cat(F_).numpy(), torch.cat(Lg).numpy(), torch.cat(Y).numpy()
    return feats, logits, logits.max(1), logits.argmax(1), y


# ---------------- resume ----------------
start, best_acc, log = 0, -1.0, []
if (out / "last.pt").exists():
    ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); crit.load_state_dict(ck["crit"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    start, best_acc, log = ck["epoch"] + 1, ck["best_acc"], ck["log"]
    torch.set_rng_state(ck["rng"].cpu())
    print(f"[resume] from epoch {start}, best val acc {best_acc:.4f}")

# ---------------- train ----------------
for ep in range(start, args.epochs):
    model.train(); crit.train()
    t0, tot, n = time.time(), 0.0, 0
    for x, y in train_loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        _, (_, loss) = forward(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
    sched.step()
    _, _, _, pv, yv = infer(val_loader)
    acc = float((pv == yv).mean())
    rec = {"epoch": ep, "train_loss": tot / n, "val_acc": acc, "radius": float(crit.radius.item()),
           "sec": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    log.append(rec)
    print(rec)
    if acc > best_acc:
        best_acc = acc
        atomic_save({"model": model.state_dict(), "crit": crit.state_dict(), "classes": classes,
                     "args": vars(args), "epoch": ep, "val_acc": acc}, out / "best.pt")
    atomic_save({"model": model.state_dict(), "crit": crit.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep,
                 "best_acc": best_acc, "log": log, "rng": torch.get_rng_state()}, out / "last.pt")
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)

# ---------------- outputs + evaluation (best checkpoint) ----------------
best = torch.load(out / "best.pt", map_location=dev, weights_only=False)
model.load_state_dict(best["model"]); crit.load_state_dict(best["crit"])
res = {}
for s in ["val", "test_known", "test_unknown"]:
    feats, logits, score, pred, y = infer(L(s, False))
    res[s] = dict(score=score, pred=pred, y=y)
    np.savez_compressed(out / f"outputs_{s}.npz", features=feats, logits=logits, score=score,
                        pred=pred, label=y, path=parts[s]["path"].to_numpy())
tau = threshold_at_tpr(res["val"]["score"], args.tpr)
k, u = res["test_known"], res["test_unknown"]
m = evaluate(k["score"], k["pred"], k["y"], u["score"], u["pred"], tau, len(classes))
du = parts["test_unknown"]
m["detection_by_site_type"] = rejection_rate_by(u["score"], tau, du["site_type"])
m["detection_by_unknown_group"] = rejection_rate_by(u["score"], tau, du["unknown_group"])
m["detection_by_class"] = rejection_rate_by(u["score"], tau, du["model_code"])
m.update(best_epoch=int(best["epoch"]), best_val_acc=float(best["val_acc"]), classes=classes,
         args=vars(args), n={s: len(d) for s, d in parts.items()})
best["tau"] = tau
atomic_save(best, out / "best.pt")
(out / "metrics.json").write_text(json.dumps(m, indent=2))
print(json.dumps({k_: v for k_, v in m.items() if not isinstance(v, (dict, list))}, indent=2))
(out / "last.pt").unlink(missing_ok=True)  # free Drive space; best.pt is kept
(out / "DONE").touch()
