# BTODV-IL: Class-Incremental CRDM Fault Diagnosis

This repository provides a reference implementation for **BTODV-IL**, a dual-threshold voting spiking framework for noise-robust class-incremental fault diagnosis of nuclear control rod drive mechanisms (CRDMs).

## Included files

- `btodvil/models/btodv_snn.py`: BTODV-SNN backbone with theta-oscillatory virtual membrane dynamics, dual-threshold voting spike generation, and adaptive threshold regulation.
- `btodvil/continual.py`: class-incremental learning utilities, including classifier expansion, logits distillation, and exemplar replay.
- `scripts/train_incremental_cil_snr.py`: a compact example script for stage-wise training and SNR-controlled disturbance evaluation.

## Data availability

The private CRDM dataset used in the paper is not released because of confidentiality restrictions. The example script assumes the following class