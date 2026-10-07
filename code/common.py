"""Shared data / model utilities (reused by ARPL and the baselines later)."""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_H, IMG_W = 128, 256  # native crop size (width 256, height 128); no square resizing


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(train, rot=15):
    t = [T.Resize((IMG_H, IMG_W))]  # safety only; all images are already 128x256
    if train:
        t += [T.RandomHorizontalFlip(), T.RandomRotation(rot, fill=0),
              T.ColorJitter(brightness=0.2, contrast=0.2)]
    t += [T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(t)


class ImplantDataset(Dataset):
    def __init__(self, df, root, class_to_idx, transform):
        self.root = Path(root)
        self.paths = df["path"].tolist()
        self.labels = [class_to_idx.get(m, -1) for m in df["model_code"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            img = im.convert("L")
        return self.transform(img), self.labels[i]


def load_split(split_csv, limit=0, seed=0):
    df = pd.read_csv(split_csv, dtype=str)
    classes = sorted(df.loc[df.role == "known", "model_code"].unique())
    parts = {s: df[df.split == s].reset_index(drop=True)
             for s in ["train", "val", "test_known", "test_unknown"]}
    if limit:  # smoke test: small random subset of every split
        parts = {s: d.sample(min(len(d), limit), random_state=seed).reset_index(drop=True)
                 for s, d in parts.items()}
    return parts, classes


def make_loader(df, root, class_to_idx, train, batch_size, workers):
    ds = ImplantDataset(df, root, class_to_idx, build_transform(train))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=workers > 0)


def build_backbone(name, pretrained=True):
    """timm backbone returning pooled features (classifier removed)."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, model.num_features


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
