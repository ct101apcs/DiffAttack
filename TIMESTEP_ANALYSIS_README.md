# Per-Timestep Loss Analysis for DiffAttack

## Overview

This implementation adds comprehensive per-timestep loss breakdown capabilities to the DiffAttack framework. It allows you to understand which diffusion timesteps contribute most to the attack success, providing insights into the attack mechanism.

## Features

### 1. **Timestep Loss Tracking**
- Tracks self-attention loss for each diffusion timestep
- Tracks cross-attention variance loss per timestep
- Tracks attack loss per timestep
- Records loss contributions across all optimization iterations

### 2. **Visualization Suite**

#### A. Loss per Timestep at Key Iterations
**File**: `loss_per_timestep.png`
- Line plots showing loss values across timesteps
- Displays snapshots at different iterations (start, middle, end)
- Separate subplots for:
  - Self-Attention Loss
  - Cross-Attention Variance Loss
  - Attack Loss
  - Total Loss

#### B. Heatmap: Timestep vs Iteration
**File**: `loss_heatmap.png`
- Color-coded heatmaps showing loss evolution
- X-axis: Iteration number
- Y-axis: Timestep index
- Individual heatmaps for each loss type
- Helps identify which timesteps are most critical throughout optimization

#### C. Average Loss per Timestep
**File**: `average_loss_per_timestep.png`
- Shows mean ± std deviation across all iterations
- Identifies consistently important timesteps
- Helps understand which diffusion stages matter most

#### D. Statistical Summary
**File**: `timestep_statistics.txt`
- Numerical breakdown of per-timestep contributions
- Identifies max/min contributing timesteps
- Compares to uniform distribution baseline

## Usage

### Running with Timestep Analysis

```bash
# Standard attack with timestep analysis (enabled by default)
python main.py \
    --images_root demo/images \
    --label_path demo/labels.txt \
    --save_dir output \
    --diffusion_steps 20 \
    --start_step 15 \
    --iterations 30 \
    --model_name inception

# Results will be saved to:
# - output/timestep_analysis/loss_per_timestep.png
# - output/timestep_analysis/loss_heatmap.png
# - output/timestep_analysis/average_loss_per_timestep.png
# - output/timestep_analysis/timestep_statistics.txt
```

### Testing the Implementation

```bash
# Run unit tests
python test_timestep_analysis.py

# Run quick integration test with single image
python test_single_image.py
```

## Implementation Details

### Modified Files

1. **attentionControl.py**
   - Added `timestep_loss_details` list to track per-timestep losses
   - Added `current_iteration` and `current_timestep_index` tracking
   - Modified `forward()` to record timestep-specific loss contributions
   - Updated `reset()` to clear timestep tracking

2. **diff_latent_attack.py**
   - Added `all_iteration_losses` dictionary to collect losses
   - Modified main optimization loop to track per-timestep losses
   - Added visualization generation before function return
   - Stores timestep details after each iteration

3. **utils.py**
   - Added `print_timestep_statistics()` for numerical summary
   - Added `plot_loss_per_timestep_iterations()` for line plots
   - Added `plot_loss_heatmap_timestep_vs_iteration()` for heatmaps
   - Added `plot_average_loss_per_timestep()` for averaged plots

### Data Structure

```python
all_iteration_losses = {
    'attack_loss': [],           # [num_iterations]
    'cross_attn_loss': [],       # [num_iterations]
    'self_attn_loss': [],        # [num_iterations]
    'total_loss': [],            # [num_iterations]
    'timestep_details': []       # [num_iterations][num_timesteps]
}

# Each entry in timestep_details:
{
    'timestep': int,        # Timestep index
    'iteration': int,       # Iteration number
    'loss': float,          # Loss value
    'loss_type': str        # 'self_attn', 'cross_attn', or 'attack'
}
```

## Interpreting Results

### Loss per Timestep Plot
- **Early timesteps (low indices)**: Control high-level semantic features
- **Late timesteps (high indices)**: Control fine-grained details
- **Peak losses**: Indicate timesteps where attack has strongest effect

### Heatmap Analysis
- **Bright regions**: Timesteps consistently contributing high loss
- **Dark regions**: Timesteps with minimal impact
- **Patterns across iterations**:
  - Horizontal bands → specific timesteps always important
  - Vertical bands → certain iterations affect all timesteps
  - Diagonal patterns → temporal correlations

### Average Loss Insights
- **High mean + low std**: Consistently important timestep
- **High mean + high std**: Important but variable across iterations
- **Low mean**: Timestep with minimal attack contribution
- **Compare to uniform baseline**: Identifies over/under-utilized timesteps

## Example Interpretations

### Scenario 1: Late-Timestep Dominance
```
Timestep 15-19: High average loss
Timestep 0-5: Low average loss
```
**Interpretation**: Attack primarily modifies fine-grained details, preserving semantic content. Good for imperceptibility.

### Scenario 2: Early-Timestep Dominance
```
Timestep 0-5: High average loss
Timestep 15-19: Low average loss
```
**Interpretation**: Attack modifies high-level features. May be more transferable but potentially more noticeable.

### Scenario 3: Uniform Distribution
```
All timesteps: Similar average loss (~33.3% each)
```
**Interpretation**: Attack utilizes full diffusion process. Balanced approach.

## Troubleshooting

### Missing Visualizations
**Problem**: PNG files not generated
**Solution**: 
```bash
pip install matplotlib seaborn
```

### Memory Issues with Many Images
**Problem**: Out of memory when processing large datasets
**Solution**: Process in batches or reduce `--diffusion_steps`

### No Timestep Statistics File
**Problem**: `timestep_statistics.txt` not created
**Solution**: Check that `--start_step < --diffusion_steps`

## Advanced Usage

### Analyzing Specific Loss Types

To focus on specific loss components, modify the visualization functions:

```python
# In utils.py
def plot_loss_per_timestep_iterations(all_losses, save_dir, loss_types=['self_attn_loss']):
    # Only plot specified loss types
    ...
```

### Custom Timestep Ranges

To analyze specific timestep ranges:

```python
# Filter timestep_details for range [10, 15]
filtered_details = [
    [d for d in iteration if 10 <= d['timestep'] <= 15]
    for iteration in all_losses['timestep_details']
]
```

## Citation

If you use this timestep analysis functionality, please cite the original DiffAttack paper and acknowledge this extension:

```bibtex
@article{diffattack2024,
  title={DiffAttack: Adversarial Attacks via Diffusion Models},
  author={...},
  journal={...},
  year={2024}
}
```

## Future Enhancements

- [ ] Real-time timestep loss monitoring during training
- [ ] Timestep importance ranking for attack speedup
- [ ] Automatic timestep selection based on loss distribution
- [ ] Correlation analysis between timestep losses and transferability
- [ ] Interactive visualization dashboard

## Contact

For issues or questions about timestep analysis:
- Create an issue in the repository
- Check the test scripts for usage examples
