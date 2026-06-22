"""Compact BTODV-IL training example.

This script demonstrates stage-wise class-incremental training with exemplar
replay, old-class distillation, and SNR-controlled disturbance evaluation.
It is a public reference script; private CRDM data are not included.
"""

import argparse
import copy
import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from btodvil.models.btodv_snn import BTODVSNN
from btodvil.continual import MemoryBuffer, expand_classifier, incremental_loss


STAGES: Dict[str, List[str]] = {
    "Stage1": ["normal_insert", "normal_withdraw", "slider_insert", "slider_withdraw"],
    "Stage2": ["normal_insert", "normal_withdraw", "slider_insert", "slider_withdraw", "unable_to_lift"],
    "Stage3": ["normal_insert", "normal_withdraw", "slider_insert", "slider_withdraw", "unable_to_lift", "clip_insert", "clip_withdraw"],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FolderSpikeDataset(Dataset):
    """Small placeholder dataset loader.

    Replace this class with the private CRDM loader or prepare `.npy` samples in
    class folders. Each sample should have shape [T, 1, D].
    """
    def __init__(self, root: str, classes: List[str], time_steps: int = 32, dim: int = 1024):
        self.samples = []
        self.labels = []
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            folder = os.path.join(root, c)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if name.endswith(".npy"):
                    self.samples.append(os.path.join(folder, name))
                    self.labels.append(self.class_to_idx[c])
        self.time_steps = time_steps
        self.dim = dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.load(self.samples[idx]).astype("float32")
        if x.ndim == 1:
            x = np.tile(x[None, None, :], (self.time_steps, 1, 1))
        return torch.tensor(x).float(), torch.tensor(self.labels[idx]).long()


def add_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    if np.isinf(snr_db):
        return x
    power = x.pow(2).mean(dim=tuple(range(1, x.ndim)), keepdim=True)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    return x + torch.randn_like(x) * torch.sqrt(noise_power + 1e-12)


@torch.no_grad()
def evaluate(model, loader, device, snr_db=float("inf")):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = add_awgn(x, snr_db)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


def train_one_stage(model, teacher, train_loader, mem_loader, device, args, old_classes: int):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        mem_iter = iter(mem_loader) if mem_loader is not None else None
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(x)
            loss = incremental_loss(logits, y, teacher_logits, old_classes, args.alpha_kd, args.temperature)
            if mem_iter is not None:
                try:
                    xm, ym = next(mem_iter)
                except StopIteration:
                    mem_iter = iter(mem_loader)
                    xm, ym = next(mem_iter)
                xm, ym = xm.to(device), ym.to(device)
                lm = model(xm)
                tm = teacher(xm) if teacher is not None else None
                loss = 0.5 * loss + 0.5 * incremental_loss(lm, ym, tm, old_classes, args.alpha_kd, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/CRDM_NPY")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--time_steps", type=int, default=32)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--alpha_kd", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--m_per_class", type=int, default=20)
    p.add_argument("--snr_levels", nargs="+", type=float, default=[float("inf"), 10.0, 4.0, 0.0, -4.0])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, teacher = None, None
    memory = MemoryBuffer(args.m_per_class)
    old_classes = 0

    for stage_name, classes in STAGES.items():
        ds = FolderSpikeDataset(args.data_dir, classes, args.time_steps, args.dim)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        if model is None:
            model = BTODVSNN(num_classes=len(classes), time_steps=args.time_steps).to(device)
        else:
            teacher = copy.deepcopy(model).eval().to(device)
            for param in teacher.parameters():
                param.requires_grad = False
            expand_classifier(model, len(classes))
        mem_loader = memory.build_loader(args.batch_size)
        train_one_stage(model, teacher, loader, mem_loader, device, args, old_classes)
        for x, y in loader:
            memory.add_batch(x, y)
        old_classes = len(classes)
        scores = {str(s): evaluate(model, eval_loader, device, s) for s in args.snr_levels}
        print(stage_name, scores)


if __name__ == "__main__":
    main()
