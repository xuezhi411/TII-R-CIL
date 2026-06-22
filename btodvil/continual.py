"""Minimal class-incremental learning utilities for BTODV-IL."""

from typing import Dict, List, Optional, Tuple
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def expand_classifier(model: nn.Module, new_num_classes: int, head_name: str = "classifier") -> None:
    """Expand a linear classifier and preserve old-class weights."""
    head = getattr(model, head_name)
    if new_num_classes <= head.out_features:
        return
    new_head = nn.Linear(head.in_features, new_num_classes, bias=head.bias is not None).to(head.weight.device)
    with torch.no_grad():
        new_head.weight[: head.out_features].copy_(head.weight)
        if head.bias is not None:
            new_head.bias[: head.out_features].copy_(head.bias)
        nn.init.kaiming_uniform_(new_head.weight[head.out_features :], a=math.sqrt(5))
    setattr(model, head_name, new_head)


class LogitsDistillation(nn.Module):
    """Logits distillation over old classes only."""
    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, old_classes: int) -> torch.Tensor:
        if old_classes <= 0:
            return student_logits.new_zeros(())
        t = self.temperature
        s = F.log_softmax(student_logits[:, :old_classes] / t, dim=1)
        q = F.softmax(teacher_logits[:, :old_classes] / t, dim=1)
        return F.kl_div(s, q, reduction="batchmean") * (t * t)


class MemoryBuffer:
    """Per-class replay buffer storing CPU tensors."""
    def __init__(self, m_per_class: int = 20):
        self.m_per_class = int(m_per_class)
        self.data: Dict[int, List[Tuple[torch.Tensor, int]]] = {}

    def __len__(self):
        return sum(len(v) for v in self.data.values())

    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().cpu()
        y = y.detach().cpu()
        for xi, yi in zip(x, y):
            c = int(yi.item())
            self.data.setdefault(c, []).append((xi.clone(), c))
            if len(self.data[c]) > self.m_per_class:
                self.data[c].pop(0)

    def build_loader(self, batch_size: int = 64) -> Optional[DataLoader]:
        xs, ys = [], []
        for items in self.data.values():
            for x, y in items:
                xs.append(x)
                ys.append(y)
        if not xs:
            return None
        xs = torch.stack(xs)
        ys = torch.tensor(ys, dtype=torch.long)
        counts = torch.bincount(ys)
        weights = (counts.sum().float() / (counts.float() + 1e-6))[ys]
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(ys), replacement=True)
        return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, sampler=sampler)


def incremental_loss(student_logits, labels, teacher_logits=None, old_classes: int = 0, alpha_kd: float = 0.5, temperature: float = 3.0):
    ce = F.cross_entropy(student_logits, labels)
    if teacher_logits is None or old_classes <= 0:
        return ce
    kd = LogitsDistillation(temperature)(student_logits, teacher_logits, old_classes)
    return (1.0 - alpha_kd) * ce + alpha_kd * kd
