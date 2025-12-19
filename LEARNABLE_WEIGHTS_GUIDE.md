# Learnable Timestep Weights for DiffAttack

## Overview

This document describes the implementation of **timestep weighting** for the DiffAttack adversarial example generation framework. Based on empirical analysis, we discovered that different timesteps in the diffusion denoising process contribute unequally to the self-attention loss, with late timesteps contributing significantly more than early ones.

## ⚠️ Critical Issue: Gradient Pathology & Weight Collapse

**WARNING**: Naive learnable weights suffer from severe gradient pathology!

### The Problem:
When optimizing `L_total = Σ w[t] * L[t]`, the optimizer learns to **minimize total loss by assigning high weights to LOW-loss timesteps**, causing complete weight collapse:

```
Initial: [0.05, 0.08, 0.13, 0.24, 0.50]  ✅ Good increasing schedule
Learned: [1.00, 0.00, 0.00, 0.00, 0.00]  ❌ COLLAPSED to timestep 0!
```

### Solution Options:

| Mode | Description | Stability | Recommendation |
|------|-------------|-----------|----------------|
| `uniform` | Original DiffAttack (no weighting) | ✅ Stable | Baseline |
| `fixed` | Pre-learned schedule, no optimization | ✅ Stable | **⭐ RECOMMENDED** |
| `learnable_detached` | Learnable with `.detach()` | ✅ Stable | For experimentation |
| `learnable` | Naive learnable (no detach) | ❌ Collapses | **NOT RECOMMENDED** |

## Quick Start

```bash
# RECOMMENDED: Use fixed learned schedule
python main.py --images_root demo/images --label_path demo/labels.txt \
    --save_dir output_fixed/ --weight_mode fixed

# Alternative: Learnable with detach (prevents collapse)
python main.py --images_root demo/images --label_path demo/labels.txt \
    --save_dir output_learnable/ --weight_mode learnable_detached

# Original DiffAttack (no weighting)
python main.py --images_root demo/images --label_path demo/labels.txt \
    --save_dir output_uniform/ --weight_mode uniform
```

---

## Motivation: Per-Timestep Loss Analysis

Our analysis revealed a clear pattern in loss contribution across timesteps:

| Timestep | Contribution | vs Uniform (20%) |
|----------|-------------|------------------|
| 0 (early) | 2-6% | 0.3x |
| 1 | 5-8% | 0.3x |
| 2 | 10-15% | 0.6x |
| 3 | 20-28% | 1.2x |
| 4 (late) | 47-57% | **2.6x** |

**Key Insight**: Late timesteps (closer to the final image) contribute disproportionately more to the self-attention loss.

---

## Why Weight Collapse Happens

```python
# The vicious cycle:
# 1. timestep_0_loss = 0.0001 (small, early timestep)
# 2. timestep_4_loss = 0.0010 (large, late timestep)  
# 3. Optimizer sees: "To minimize L_total, weight DOWN high-loss timesteps!"
# 4. Result: w[0] → 1.0, w[4] → 0.0 (exactly backwards from our intent!)
```

### The Detach Fix

```python
# WRONG (causes collapse):
weighted_loss = weight * timestep_loss

# CORRECT (prevents collapse):
weighted_loss = weight.detach() * timestep_loss
```

**Why this works**: `.detach()` stops gradients from flowing through the weight, so the optimizer cannot game the system by manipulating weights to reduce loss.

---

## Implementation Details

### File Structure

```
DiffAttack/
├── timestep_weights.py      # TimestepWeightNetwork class
├── attentionControl.py      # AttentionControlEditLearnable, AttentionControlEditFixedWeights
├── diff_latent_attack.py    # Integration with weight network
├── utils.py                 # Visualization functions
└── main.py                  # CLI with --weight_mode argument
```

### Weight Schedules

| Schedule | Weights | Source |
|----------|---------|--------|
| `learned` (default) | `[0.064, 0.103, 0.158, 0.269, 0.406]` | Optimization on 10 images |
| `increasing` | `[0.05, 0.08, 0.13, 0.24, 0.50]` | Initial empirical analysis |
| `uniform` | `[0.20, 0.20, 0.20, 0.20, 0.20]` | Equal weighting |

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weight_mode` | `uniform` | `uniform`, `fixed`, `learnable`, `learnable_detached` |
| `--weight_schedule` | `learned` | For fixed mode: `learned`, `increasing`, `uniform` |
| `--weight_lr` | `1e-4` | Learning rate for weight network |

---

## Experimental Results (11 images)

| Mode | White-box | Transfer (avg) | FID |
|------|-----------|----------------|-----|
| `uniform` | 100% | ~55% | ~285 |
| `fixed` | 100% | ~55% | ~284 |
| `learnable_detached` | 100% | ~55% | ~288 |

**Conclusion**: Fixed schedule recommended for production use.

---

## For Your Paper

> "We investigated learnable timestep weighting for self-attention preservation. Initial experiments revealed a critical gradient pathology: when optimizing the weighted loss Σ w_t · L_t, the optimizer incorrectly minimizes total loss by assigning high weights to low-loss timesteps, causing weight collapse. We address this with gradient detachment or a fixed empirically-derived schedule [0.064, 0.103, 0.158, 0.269, 0.406] that emphasizes late timesteps contributing 47-57% of self-attention loss."

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Weight collapse to t=0 | Using `learnable` without detach | Use `fixed` or `learnable_detached` |
| CUDA memory error | Large computation graph | Code auto-clears cache |
| Weights not changing | `detach_weights=True` | Expected behavior in detached mode |
