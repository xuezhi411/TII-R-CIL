"""BTODV-SNN reference model.

This file provides a compact implementation of the core mechanism used in
BTODV-IL: theta-oscillatory virtual membrane dynamics, dual-threshold voting
spike generation, and adaptive threshold regulation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ATanSurrogate(torch.autograd.Function):
    """Binary