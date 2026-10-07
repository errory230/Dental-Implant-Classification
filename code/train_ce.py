"""
Train the cross-entropy (CE) baseline: same backbone, data, augmentation and schedule as ARPL,
with a linear classifier instead of reciprocal points. One CE model per seed serves all post-hoc
baselines (MSP, MLS, Energy, ODIN, OpenMax), which are scored later by eval_baselines.py.

Output per seed (<out_dir>/ce/seed{k}/):
  best.pt, log.csv, DONE
  outputs_train.npz                      logits/pred/label (no augmentation; used to fit OpenMax)
  outputs_{val,test_known,test_unknown}.npz   logits, features(fp16), pred, label, path, odin_score

Usage (Colab):
  !python train_ce.py --split_csv ".../splits/seed0.csv" --data_root /content/data/panorama \
      --out_dir ".../runs" --seed 0
  Smoke test: add --limit 256 --epochs 1 --run_name smoke_ce
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import atomic_save, build_backbone, load_split, make_loader, seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--split_csv", required=True)
ap.add_argument("--data_root", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--run_name", default="ce")
ap.add_argument("--backbone", default="convnext_small.fb_in22k")
ap.add_argument("--pretrained", type=int, default=1)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--weight_decay", type=float, default=0.05)
ap.add_argument("--head_lr", type=float, default=1e-3, help="lr for the new linear head (same as ARPL crit_lr)")
ap.add_argument("--odin_T", type=float, default=1000.0)
ap.add_argument("--odin_eps", type=float, default=0.0014)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--amp", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
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

parts, classes = load_split(args.split_csv, args.limit, args.seed)
c2i = {c: i for i, c in enumerate(classes)}
print(f"classes ({len(classes)}): {classes}")
print({k: len(v) for k, v in parts.items()})
L = lambda s, train: make_loader(parts[s], args.data_root, c2i, train, args.batch_size, args.workers)
train_loader, val_loader = L("train", True), L("val", False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, d = build_backbone(args.backbone, bool(args.pretrained))
        self.head = nn.Linear(d, len(classes))

    def forward(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            f = self.backbone(x)
        f = f.float()
        return f, self.head(f)


net = Net().to(dev)
opt = torch.optim.AdamW([
    {"params": net.backbone.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
    {"params": net.head.parameters(), "lr": args.head_lr, "weight_decay": 0.0}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


@torch.no_grad()
def infer(loader, keep_feats=True):
    net.eval()
    F_, Lg, Y = [], [], []
    for x, y in loader:
        f, logits = net(x.to(dev, non_blocking=True))
        if keep_feats:
            F_.append(f.half().cpu())
        Lg.append(logits.cpu()); Y.append(y)
    logits = torch.cat(Lg).numpy()
    feats = torch.cat(F_).numpy() if keep_feats else None
    return feats, logits, logits.argmax(1), torch.cat(Y).numpy()


def odin_scores(loader):
    """ODIN (Liang et al., 2018): temperature scaling + input perturbation, fixed T and eps
    (not tuned on unknown data). Perturbation is applied in the normalized input space."""
    net.eval()
    out_ = []
    for x, _ in loader:
        x = x.to(dev).requires_grad_(True)
        _, logits = net(x)
        logits = logits / args.odin_T
        loss = F.cross_entropy(logits, logits.argmax(1))
        grad, = torch.autograd.grad(loss, x)
        x2 = x.detach() - args.odin_eps * grad.sign()
        with torch.no_grad():
            _, l2 = net(x2)
            out_.append(F.softmax(l2 / args.odin_T, 1).max(1).values.cpu())
    return torch.cat(out_).numpy()


start, best_acc, log = 0, -1.0, []
if (out / "last.pt").exists():
    ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    start, best_acc, log = ck["epoch"] + 1, ck["best_acc"], ck["log"]
    torch.set_rng_state(ck["rng"].cpu())
    print(f"[resume] from epoch {start}, best val acc {best_acc:.4f}")

for ep in range(start, args.epochs):
    net.train()
    t0, tot, n = time.time(), 0.0, 0
    for x, y in train_loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        _, logits = net(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
    sched.step()
    _, _, pv, yv = infer(val_loader, keep_feats=False)
    acc = float((pv == yv).mean())
    rec = {"epoch": ep, "train_loss": tot / n, "val_acc": acc, "sec": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    log.append(rec)
    print(rec)
    if acc > best_acc:
        best_acc = acc
        atomic_save({"net": net.state_dict(), "classes": classes, "args": vars(args),
                     "epoch": ep, "val_acc": acc}, out / "best.pt")
    atomic_save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": ep, "best_acc": best_acc, "log": log,
                 "rng": torch.get_rng_state()}, out / "last.pt")
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)

best = torch.load(out / "best.pt", map_location=dev, weights_only=False)
net.load_state_dict(best["net"])
_, logits, pred, y = infer(L("train", False), keep_feats=False)
np.savez_compressed(out / "outputs_train.npz", logits=logits, pred=pred, label=y)
for s in ["val", "test_known", "test_unknown"]:
    loader = L(s, False)
    feats, logits, pred, y = infer(loader)
    odin = odin_scores(loader)
    np.savez_compressed(out / f"outputs_{s}.npz", features=feats, logits=logits, pred=pred, label=y,
                        path=parts[s]["path"].to_numpy(), odin_score=odin)
    print(f"saved outputs_{s}.npz")
(out / "last.pt").unlink(missing_ok=True)
(out / "DONE").touch()
print(f"done. best epoch {best['epoch']}, val acc {best['val_acc']:.4f}")
