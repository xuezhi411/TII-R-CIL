"""BTODV-SNN reference model.

Compact implementation of theta-oscillatory dual-threshold voting spiking
neurons and a lightweight 1-D SNN backbone for BTODV-IL experiments.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn


class ATanSurrogate(torch.autograd.Function):
    """Binary spike with atan surrogate gradient."""
    @staticmethod
    def forward(ctx, x, alpha: float = 2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = grad_output * (alpha / 2.0) / (1.0 + (math.pi * alpha * x / 2.0) ** 2)
        return grad, None


@dataclass
class VotingLIFHyperparameters:
    thresh_base: float = 1.0
    virtual_thresh_ratio: float = 0.85
    voting_threshold: int = 3
    oscillation_amplitude: float = 0.10
    theta_frequency: float = 2 * math.pi / 8
    beta: float = 0.90
    alpha: float = 0.80
    adapt_rate: float = 0.01
    target_spike_rate: float = 0.25


class BTODVNeuron(nn.Module):
    """Theta-oscillatory dual-threshold voting LIF neuron.

    The real-threshold path fires for strong instantaneous responses. The
    virtual-threshold path records repeated subthreshold responses and emits a
    vote-triggered spike when the counter reaches K. A spike-rate feedback term
    adjusts the threshold to suppress disturbance-induced over-firing.
    """

    def __init__(self, hp: Optional[VotingLIFHyperparameters] = None, learnable: bool = True):
        super().__init__()
        self.hp = hp or VotingLIFHyperparameters()
        wrap = nn.Parameter if learnable else lambda x: x
        self.thresh = wrap(torch.tensor(float(self.hp.thresh_base)))
        self.virtual_ratio = wrap(torch.tensor(float(self.hp.virtual_thresh_ratio)))
        self.osc_amp = wrap(torch.tensor(float(self.hp.oscillation_amplitude)))
        self.theta_freq = wrap(torch.tensor(float(self.hp.theta_frequency)))
        self.beta = wrap(torch.tensor(float(self.hp.beta)))
        self.alpha = wrap(torch.tensor(float(self.hp.alpha)))
        self.voting_threshold = int(self.hp.voting_threshold)
        self.register_buffer("thresh_adapt", torch.tensor(0.0))
        self.vote_count = None

    def reset_state(self):
        self.vote_count = None

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None,
                syn: Optional[torch.Tensor] = None, t: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mem is None:
            mem = torch.zeros_like(x)
        if syn is None:
            syn = torch.zeros_like(x)
        if self.vote_count is None or self.vote_count.shape != x.shape:
            self.vote_count = torch.zeros_like(x)

        syn = self.alpha.clamp(0.0, 1.0) * syn + x
        mem = self.beta.clamp(0.0, 1.0) * mem + syn

        theta = self.osc_amp * torch.sin(self.theta_freq * t)
        v_virtual = mem + theta
        v_real = self.thresh + self.thresh_adapt
        v_vir = v_real * self.virtual_ratio.clamp(0.5, 0.99)

        virtual_event = (v_virtual >= v_vir).float()
        self.vote_count = self.vote_count + virtual_event
        vote_spike = (self.vote_count >= self.voting_threshold).float()
        real_spike = ATanSurrogate.apply(mem - v_real, 2.0)
        spike = torch.maximum(real_spike, vote_spike)

        spike_mask = spike > 0
        self.vote_count = torch.where(spike_mask, torch.zeros_like(self.vote_count), self.vote_count)
        mem = torch.where(spike_mask, mem - v_real, mem)

        with torch.no_grad():
            rate = spike.mean()
            self.thresh_adapt.mul_(0.99).add_(self.hp.adapt_rate * (rate - self.hp.target_spike_rate))
            self.thresh_adapt.clamp_(-0.3, 0.3)
        return spike, mem, syn


class SpikeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hp: Optional[VotingLIFHyperparameters] = None):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.neuron = BTODVNeuron(hp)

    def reset_state(self):
        self.neuron.reset_state()

    def forward(self, x: torch.Tensor, state, t: int):
        y = self.bn(self.conv(x))
        b, c, l = y.shape
        y2 = y.reshape(b, c * l)
        mem, syn = (None, None) if state is None else state
        spk, mem, syn = self.neuron(y2, mem, syn, t)
        return spk.reshape(b, c, l), (mem, syn)


class BTODVSNN(nn.Module):
    """Lightweight BTODV-SNN classifier.

    Input shape: [B, T, 1, D]. Output shape: [B, num_classes].
    """

    def __init__(self, num_classes: int, time_steps: int = 32, base_channels: int = 32,
                 hp: Optional[VotingLIFHyperparameters] = None):
        super().__init__()
        self.time_steps = time_steps
        self.block1 = SpikeConvBlock(1, base_channels, hp)
        self.block2 = SpikeConvBlock(base_channels, base_channels * 2, hp)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_channels * 2, num_classes)

    def reset_state(self):
        self.block1.reset_state()
        self.block2.reset_state()

    def forward(self, x: torch.Tensor):
        self.reset_state()
        s1 = s2 = None
        feats = []
        steps = min(x.shape[1], self.time_steps)
        for t in range(steps):
            y, s1 = self.block1(x[:, t], s1, t)
            y, s2 = self.block2(y, s2, t)
            feats.append(self.pool(y).squeeze(-1))
        h = torch.stack(feats, dim=1).mean(dim=1)
        return self.classifier(h)


def expand_classifier(model: nn.Module, new_num_classes: int):
    """Expand final classifier while preserving old rows."""
    old = model.classifier
    if new_num_classes <= old.out_features:
        return
    new = nn.Linear(old.in_features, new_num_classes).to(old.weight.device)
    with torch.no_grad():
        new.weight[: old.out_features].copy_(old.weight)
        new.bias[: old.out_features].copy_(old.bias)
    model.classifier = new
