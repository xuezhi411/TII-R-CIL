# BTODV-IL: Class-Incremental CRDM Fault Diagnosis

This repository provides a reference implementation for **BTODV-IL**, a bio-inspired dual-threshold voting spiking framework for noise-robust class-incremental fault diagnosis of nuclear control rod drive mechanisms (CRDMs).

## What is included

- `btodvil/models/btodv_snn.py`: BTODV-SNN backbone with theta-oscillatory virtual membrane dynamics, dual-threshold voting spike generation, and adaptive threshold regulation.
- `btodvil/continual.py`: helper functions for class-incremental learning, including classifier expansion, logits distillation, and