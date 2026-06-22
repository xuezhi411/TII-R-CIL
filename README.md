# BTODV-IL: Class-Incremental CRDM Fault Diagnosis

This repository provides a compact public reference implementation for **BTODV-IL**, a dual-threshold voting spiking framework for noise-robust class-incremental fault diagnosis of nuclear control rod drive mechanisms (CRDMs).

## Included files

- `btodvil/models/btodv_snn.py`: BTODV-SNN reference model with theta-oscillatory virtual membrane dynamics, dual-threshold voting spike generation, and adaptive threshold regulation.
- `btodvil/continual.py`: class-incremental learning utilities, including classifier expansion, logits distillation, and exemplar replay.
- `scripts/train_incremental_cil_snr.py`: compact example script for stage-wise training and SNR-controlled disturbance evaluation.

## Data availability

The private CRDM dataset used in the paper is not released because of confidentiality restrictions. The public script is a reference implementation. Users can adapt it to their own data with sample tensors saved as `.npy` files in class folders.

Example layout:

```text
data/CRDM_NPY/
  normal_insert/*.npy
  normal_withdraw/*.npy
  slider_insert/*.npy
  slider_withdraw/*.npy
  unable_to_lift/*.npy
  clip_insert/*.npy
  clip_withdraw/*.npy
```

Each sample should have shape `[T, 1, D]`, for example `[32, 1, 1024]`.

## Installation

```bash
pip install -r requirements.txt
```

## Example usage

```bash
python scripts/train_incremental_cil_snr.py \
  --data_dir ./data/CRDM_NPY \
  --time_steps 32 \
  --dim 1024 \
  --snr_levels inf 10 4 0 -4
```

## Note on disturbance evaluation

The SNR evaluation is a controlled disturbance test. Additive noise at different SNR levels is used to evaluate low-SNR robustness. It should not be interpreted as a full physical emulation of hot-state reactor operation.

## Citation

If this repository is useful, please cite the corresponding paper once it is available.
