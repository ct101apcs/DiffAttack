# Concise Prompt for Agent: Per-Timestep Loss Analysis

```
Implement per-timestep loss breakdown following your executive plan. Here's what I need:

## MODIFICATIONS REQUIRED

### 1. attentionControl.py (3 changes)
- **Constructor (lines 82-87)**: Add `self.timestep_loss_details = []` and `self.current_iteration = 0`
- **Forward method (lines 93-100)**: Before accumulating loss, store it in timestep_loss_details as a dict with keys: {timestep, iteration, loss, loss_type}
- **Reset method (lines 65-67)**: Clear timestep_loss_details

### 2. diff_latent_attack.py (4 changes)
- **Before loop (~line 470)**: Initialize `all_iteration_losses = {}` dict to store attack_loss, cross_attn_loss, self_attn_loss, total_loss per iteration
- **Start of loop (~line 490)**: Set `controller.current_iteration = iteration`
- **Cross-attn computation (~line 500)**: Track per-timestep variance in addition to total variance
- **After loop (~line 610)**: Call 3 visualization functions and save to `{output_dir}/timestep_analysis/`

### 3. utils.py (add 3 new functions)
- `plot_loss_per_timestep_iterations()`: 2x2 subplots showing loss curves for iterations [0, 10, 20, 29]
- `plot_loss_heatmap_timestep_vs_iteration()`: Single heatmap, X=timestep, Y=iteration, color=loss magnitude
- `plot_average_loss_per_timestep()`: Bar plots with error bars showing mean±std loss per timestep, include contribution %

## EXPECTED OUTPUTS

### Console Statistics:
```
========================================
PER-TIMESTEP LOSS STATISTICS
========================================
Timestep  0: Avg Loss=0.0234, Std=0.0012, Contribution=2.1%
Timestep  1: Avg Loss=0.0456, Std=0.0023, Contribution=4.3%
...
Timestep 19: Avg Loss=0.0789, Std=0.0045, Contribution=7.8%

Total timesteps: 20
Uniform weight would be: 5.00%
Max contribution: 12.3% (timestep 15)
Min contribution: 1.2% (timestep 0)
```

### File Outputs:
```
output_dir/
└── timestep_analysis/
    ├── loss_per_timestep.png          # Line plots (2x2 grid)
    ├── loss_heatmap.png                # Single heatmap
    └── average_loss_per_timestep.png   # Bar plots with error bars
```

### Visualization Details:

**Plot 1 - loss_per_timestep.png (2x2 subplots):**
- Top-left: Self-attention loss lines for 4 key iterations
- Top-right: Cross-attention variance lines for 4 key iterations  
- Bottom-left: Combined view or attack loss
- Bottom-right: Total loss
- X-axis: Timestep (0-19), Y-axis: Loss value, Legend: iteration numbers

**Plot 2 - loss_heatmap.png:**
- Single heatmap: rows=iterations (0-29), cols=timesteps (0-19)
- Color: Self-attention loss magnitude (hot/viridis colormap)
- Colorbar on right showing scale
- Should reveal if early/mid/late timesteps dominate

**Plot 3 - average_loss_per_timestep.png (2 subplots):**
- Left: Self-attention average per timestep (bars + error bars)
- Right: Cross-attention variance per timestep (bars + error bars)
- Black dashed line: uniform baseline (100/20 = 5%)
- Immediately shows which timesteps contribute most

## KEY QUESTION TO ANSWER

Looking at Plot 3, which pattern emerges?
- **Pattern A**: Timesteps 15-19 >> Timesteps 0-5 (late dominates) → Use increasing weights
- **Pattern B**: Timesteps 8-12 peak (mid dominates) → Use Gaussian weights  
- **Pattern C**: All roughly equal ±1% (uniform) → Current weighting is optimal

## VALIDATION CHECKLIST

After implementation, confirm:
- ✅ Original code runs unchanged (backward compatible)
- ✅ 3 PNG files generated in timestep_analysis/
- ✅ Console prints statistics with contribution %
- ✅ Plots have proper labels, legends, grids
- ✅ No memory leaks (use .item() when storing losses)
- ✅ <5% performance overhead

Implement in order: attentionControl.py → diff_latent_attack.py → utils.py → test with 5 images.

Show me the console statistics output and confirm all 3 PNGs are generated.
```

---

**Follow-up if agent asks for clarification:**

```
Data structure format:
- timestep_loss_details: list of {timestep: int, iteration: int, loss: float, loss_type: str}
- all_iteration_losses: dict with lists, length = num_iterations

Error handling: Use try-except, print warnings, don't crash

Style: Match existing code style, add docstrings to new functions
```

This is ~90% shorter while still providing complete specification of structure and expected outputs!