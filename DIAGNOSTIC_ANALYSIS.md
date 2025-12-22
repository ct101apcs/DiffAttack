# Diagnostic Analysis: Zero Entropy Issue

## Problem Summary
The entropy loss consistently returns 0.0 throughout all training iterations, indicating that the max-entropy regularization is not functioning as intended.

## Root Cause Analysis

Based on the code review and terminal output, the most likely causes are:

### 1. **Empty or Zero Attention Maps (Most Likely)**
The attention extraction function `get_attention_with_grad()` may be returning:
- Empty maps (no attention at the target resolution)
- All-zero tensors
- Default zero tensor when `len(maps) == 0`

**Evidence:**
- `ent: -0.00000000` exactly zero in all iterations
- The function returns `torch.zeros(res, res, 77)` when no maps are found

### 2. **Token Slicing Issue**
```python
after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]
```
If `len(true_label) <= 2`, this slice would be empty `[:, :, 1:1]` or `[:, :, 1:0]`.

**Check:** Print `len(true_label)` to verify

### 3. **Attention Store Not Updated**
The controller's `step_store` might be empty during the gradient pass because:
- Attention is not being captured during `diffusion_step()`
- Controller is reset before attention is accessed
- Wrong key names or resolution mismatch

### 4. **Numerical Issue with Uniform Distribution**
If attention is perfectly uniform across all spatial locations:
- Each pixel has probability `1/256` for 16x16
- Entropy = `-(1/256) * log(1/256) * 256 = log(256) ≈ 5.5`
- Should NOT be zero

## Diagnostic Changes Made

I've added comprehensive debugging to three key functions:

### 1. `max_entropy_loss()` - Enhanced with debug mode
```python
def max_entropy_loss(attn_map, eps=1e-10, debug=False, iteration=None):
```
**Prints every 5 iterations:**
- Shape and dimensions
- Min/Max/Mean/Sum statistics
- Whether map is all zeros
- Probability concentration
- Step-by-step entropy calculation

### 2. `get_attention_with_grad()` - Enhanced with debug mode
```python
def get_attention_with_grad(..., debug: bool = False):
```
**Prints on debug:**
- Available attention store keys
- Number of maps per key
- Shape and resolution matching
- How many maps are actually collected
- Warning if returning zeros

### 3. Main training loop - Enabled periodic debugging
```python
debug_mode = (iter_idx == 0 or iter_idx % 10 == 0)
```
Enables debug output on iteration 0, 10, 20, etc.

## How to Run Diagnostics

Run your training command again:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --save_dir maxent_demo/ --images_root demo/images --label_path demo/labels.txt --model_name resnet --cross_attn_loss_weight 100 2>&1 | tee diagnostic_output.txt
```

The debug output will show exactly what's happening at iteration 0.

## Standalone Diagnostic Tool

I've also created `diagnostic_entropy.py` which includes:
- Test cases with known entropy values
- Step-by-step entropy calculation
- Attention map analysis functions

Run it standalone:
```bash
python diagnostic_entropy.py
```

## Expected Diagnostic Output

When you run the code, look for these patterns:

### Pattern A: Empty Attention Maps
```
[DEBUG] get_attention_with_grad:
  Looking for keys: ['up_cross', 'down_cross']
  Available keys: []
  ⚠️ WARNING: No attention maps found! Returning zeros.
```
**Fix:** Attention not being captured. Check if `diffusion_step()` properly triggers attention hooks.

### Pattern B: Wrong Resolution
```
[DEBUG] get_attention_with_grad:
  Target resolution: 16x16 = 256
  Map 0: shape=[..., 64, ...], q=64, target=256
  Map 1: shape=[..., 1024, ...], q=1024, target=256
  Total maps collected: 0
```
**Fix:** Resolution mismatch. Try different `args.res // 32` values or check actual attention resolutions.

### Pattern C: Zero Attention Values
```
[DEBUG Iter 0] Attention Map Analysis:
  Shape: (16, 16, 2)
  Min/Max/Mean: 0.000000/0.000000/0.000000
  All zeros: True
```
**Fix:** Attention extraction working but values are zero. Check diffusion step implementation.

### Pattern D: Wrong Token Slicing
```
[DEBUG Iter 0] Attention Map Analysis:
  Shape: (16, 16, 0)  # Empty token dimension!
```
**Fix:** `len(true_label)` is too small. Adjust token slicing logic.

## Recommended Fixes

Once you identify the root cause from the diagnostic output:

### Fix 1: If attention maps are empty (Pattern A)
The `step_store` is likely being cleared before we access it. Try using `attention_store` instead:
```python
attn_store = getattr(controller, 'attention_store', {})  # Remove step_store fallback
```

### Fix 2: If resolution mismatch (Pattern B)
Print actual resolutions and adjust:
```python
# Before the loop, find what resolutions are actually available
for key in attn_store.keys():
    for attn in attn_store[key]:
        print(f"Available resolution: {int(np.sqrt(attn.shape[1]))}")
```

### Fix 3: If token slicing is wrong (Pattern D)
Add safeguard:
```python
token_start = min(1, len(true_label) - 1)
token_end = max(token_start + 1, len(true_label) - 1)
after_true_label_attention_map = after_attention_map[:, :, token_start:token_end]
```

### Fix 4: If attention values are actually zero (Pattern C)
This suggests the diffusion step isn't properly activating attention hooks. Check:
1. Controller is registered with the model
2. `register_attention_control()` is called
3. Hooks are not being bypassed

## Alternative: Use Gradient-Free Entropy

If the gradient computation is causing issues, you can separate gradient flow:

```python
# Keep gradients for latent but not for entropy calculation
with torch.no_grad():
    after_attention_map_detached = get_attention_with_grad(...)

# Or use the existing aggregate_attention which is gradient-free
after_attention_map = aggregate_attention(
    prompt, controller, args.res // 32, ("up", "down"), True, 1, is_cpu=False
)
after_attention_map = torch.from_numpy(after_attention_map).cuda().requires_grad_(False)
```

## Contact Points for Further Investigation

Key files to check:
1. [attentionControl.py](attentionControl.py) - Controller implementation
2. [diff_latent_attack.py](diff_latent_attack.py#L88-L117) - `get_attention_with_grad()`
3. [diff_latent_attack.py](diff_latent_attack.py#L614) - Main training loop
4. [utils.py](utils.py) - `aggregate_attention()` reference implementation

## Next Steps

1. **Run with diagnostics** - Execute the modified code
2. **Check debug output** - Look for the patterns described above
3. **Identify root cause** - Match output to patterns A, B, C, or D
4. **Apply appropriate fix** - Use the recommended fixes
5. **Verify** - Entropy should no longer be zero

If after diagnostics you're still seeing zero entropy with seemingly correct attention maps, the issue may be in the entropy formula itself or numerical precision.
