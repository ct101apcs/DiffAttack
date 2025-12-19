# Quick Start Guide: Per-Timestep Loss Analysis

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install matplotlib seaborn
# or let the test script auto-install them
```

### Step 2: Run the Test
```bash
python test_timestep_analysis.py
```

**Expected Output:**
```
✓ AttentionControlEdit initialization PASSED
✓ Loss detail tracking PASSED
✓ Reset functionality PASSED
✓ Visualization functions PASSED

ALL TESTS PASSED ✓
```

### Step 3: Run on Real Data
```bash
# Quick test with single image (~2-3 minutes)
python test_single_image.py

# Full attack on demo dataset
python main.py \
    --images_root demo/images \
    --label_path demo/labels.txt \
    --save_dir output
```

---

## 📊 What You'll Get

### Output Directory Structure
```
output/
└── timestep_analysis/
    ├── loss_per_timestep.png          # 📈 Line plots
    ├── loss_heatmap.png                # 🔥 Heatmaps
    ├── average_loss_per_timestep.png   # 📊 Averaged trends
    └── timestep_statistics.txt         # 📝 Numbers
```

---

## 📈 Visualization Guide

### 1. Loss per Timestep (loss_per_timestep.png)

**What it shows**: How losses change across diffusion timesteps at different optimization stages

**Layout**: 2x2 grid
- Top-left: Self-Attention Loss
- Top-right: Cross-Attention Variance Loss
- Bottom-left: Attack Loss
- Bottom-right: Total Loss

**How to read**:
- X-axis: Timestep index (0 = early diffusion, higher = later stages)
- Y-axis: Loss value
- Different colored lines: Different iterations (start, middle, end)
- **Peaks indicate critical timesteps** where the attack has maximum effect

**Insights**:
- If peaks are at **low timesteps** → Attack modifies high-level semantics
- If peaks are at **high timesteps** → Attack modifies fine details
- If **flat** → Attack distributed evenly across diffusion process

---

### 2. Loss Heatmap (loss_heatmap.png)

**What it shows**: Evolution of losses across both timesteps and iterations

**Layout**: 4 heatmaps stacked vertically
- Self-Attention Loss heatmap
- Cross-Attention Variance Loss heatmap
- Attack Loss heatmap
- Total Loss heatmap

**How to read**:
- X-axis: Iteration number (0 to max_iterations)
- Y-axis: Timestep index (0 to num_timesteps)
- Color: Brighter = higher loss, Darker = lower loss

**Patterns to look for**:
- **Horizontal bright bands** → Specific timesteps consistently important
- **Vertical bright bands** → Certain iterations affect all timesteps
- **Top-heavy** → Early timesteps dominate
- **Bottom-heavy** → Late timesteps dominate
- **Diagonal patterns** → Temporal correlations

**Insights**:
- Bright regions = where optimization focuses effort
- Dark regions = timesteps that could potentially be skipped
- Changes across iterations = how attack strategy evolves

---

### 3. Average Loss per Timestep (average_loss_per_timestep.png)

**What it shows**: Mean loss at each timestep with confidence intervals

**Layout**: 2x2 grid (same as plot #1)
- Each subplot shows mean ± standard deviation
- Shaded region = ±1 std
- Line = mean across all iterations

**How to read**:
- X-axis: Timestep index
- Y-axis: Average loss value
- **High mean + narrow shading** → Consistently important timestep
- **High mean + wide shading** → Important but variable
- **Low mean** → Less critical timestep

**Use this to**:
- Identify which timesteps to focus on
- Find timesteps that could be optimized differently
- Compare against uniform distribution (dashed line if shown)

---

### 4. Statistics File (timestep_statistics.txt)

**Example content**:
```
==================================================
PER-TIMESTEP LOSS STATISTICS
==================================================
Timestep  0: Avg Loss=0.015234, Std=0.003421, Contribution=15.2%
Timestep  1: Avg Loss=0.023156, Std=0.004892, Contribution=23.1%
Timestep  2: Avg Loss=0.031789, Std=0.005234, Contribution=31.8%
Timestep  3: Avg Loss=0.029821, Std=0.004123, Contribution=29.8%

Total timesteps: 4
Uniform weight would be: 25.00%
Max contribution: 31.8% (timestep 2)
Min contribution: 15.2% (timestep 0)
==================================================
```

**How to interpret**:
- **Contribution %**: How much this timestep contributes to total loss
- **Compare to uniform**: If uniform is 25%, then 31.8% is above average
- **Std**: High std means loss varies a lot across iterations

---

## 🔍 Common Patterns & Meanings

### Pattern 1: Late-Timestep Focus
```
Timesteps 15-19: High loss
Timesteps 0-5: Low loss
```
**Meaning**: Attack primarily modifies fine details
**Implication**: Good imperceptibility, may sacrifice transferability

---

### Pattern 2: Early-Timestep Focus
```
Timesteps 0-5: High loss
Timesteps 15-19: Low loss
```
**Meaning**: Attack modifies semantic features
**Implication**: May be more transferable but more noticeable

---

### Pattern 3: Uniform Distribution
```
All timesteps: ~equal contribution (±5%)
```
**Meaning**: Attack uses full diffusion process
**Implication**: Balanced approach

---

### Pattern 4: Single Peak
```
Timestep 10: Very high loss
Other timesteps: Low loss
```
**Meaning**: Attack concentrated at specific diffusion stage
**Implication**: Efficient but may be fragile

---

## 💡 Tips for Analysis

### 1. Start with the Heatmap
- Gives you the "big picture" at a glance
- Identify which timesteps light up

### 2. Check Statistics First
- Quick numerical summary
- Identifies max/min contributors immediately

### 3. Deep Dive with Line Plots
- Understand how patterns evolve during optimization
- Compare early vs late iterations

### 4. Validate with Averaged Plots
- Confirms patterns aren't just noise
- Shows statistical significance via error bars

---

## 🛠 Troubleshooting

### No plots generated?
```bash
pip install matplotlib seaborn
python test_timestep_analysis.py
```

### Plots look empty/blank?
- Check that `--start_step < --diffusion_steps`
- Ensure at least 5 iterations ran
- Verify demo images exist

### Numbers seem too small?
- This is normal! Losses are often in range 0.001-0.1
- Focus on relative comparisons, not absolute values

---

## 📞 Need Help?

1. Run tests: `python test_timestep_analysis.py`
2. Check documentation: `TIMESTEP_ANALYSIS_README.md`
3. See implementation details: `IMPLEMENTATION_SUMMARY.md`
4. Open an issue with:
   - Console output
   - Configuration used
   - Error messages (if any)

---

## ✨ Pro Tips

### Optimize Attack Based on Insights

If analysis shows timestep 15 contributes 50%:
- Consider focusing optimization there
- Adjust `--start_step` to include critical range
- Modify loss weights for that stage

### Compare Across Models

Run analysis with different `--model_name`:
```bash
for model in resnet inception vit; do
    python main.py --model_name $model --save_dir output_$model
done
```
Then compare timestep patterns across models!

### Batch Analysis

Process multiple images and aggregate statistics:
```python
# Collect all timestep_statistics.txt files
# Average across images
# Identify dataset-wide patterns
```

---

**Ready to explore?** Run `python test_single_image.py` now! 🚀
