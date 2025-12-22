"""
Diagnostic Analysis for Zero Entropy Issue
==========================================

This script analyzes why the max_entropy_loss is returning zero.

Potential causes:
1. Empty attention maps (len(maps) == 0)
2. All-zero attention values
3. Uniform attention distribution leading to zero entropy
4. Incorrect attention map dimensions/slicing
5. Numerical underflow in log computation
"""

import torch

def analyze_attention_map(attn_map, name="attention_map"):
    """Detailed diagnostic of attention map."""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")
    
    if attn_map is None:
        print("❌ Attention map is None!")
        return
    
    print(f"Shape: {attn_map.shape}")
    print(f"Device: {attn_map.device}")
    print(f"Dtype: {attn_map.dtype}")
    print(f"Requires grad: {attn_map.requires_grad}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(attn_map).any()
    has_inf = torch.isinf(attn_map).any()
    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Min: {attn_map.min().item():.8f}")
    print(f"  Max: {attn_map.max().item():.8f}")
    print(f"  Mean: {attn_map.mean().item():.8f}")
    print(f"  Std: {attn_map.std().item():.8f}")
    print(f"  Sum: {attn_map.sum().item():.8f}")
    
    # Check if all zeros
    all_zeros = (attn_map == 0).all()
    print(f"  All zeros: {all_zeros}")
    
    # Count non-zero elements
    non_zero = (attn_map != 0).sum().item()
    total = attn_map.numel()
    print(f"  Non-zero elements: {non_zero}/{total} ({100*non_zero/total:.2f}%)")
    
    return attn_map


def diagnose_max_entropy_loss(attn_map, eps=1e-10, verbose=True):
    """
    Diagnose the max_entropy_loss computation step by step.
    """
    print(f"\n{'='*60}")
    print(f"ENTROPY LOSS DIAGNOSIS")
    print(f"{'='*60}")
    
    if verbose:
        analyze_attention_map(attn_map, "Input Attention Map")
    
    b, s, t = attn_map.shape
    print(f"\nDimensions: batch={b}, spatial={s}, tokens={t}")
    
    # Step 1: Calculate sum over spatial dimension
    spatial_sum = attn_map.sum(dim=1, keepdim=True)
    print(f"\nStep 1: Spatial sum")
    print(f"  Shape: {spatial_sum.shape}")
    print(f"  Min: {spatial_sum.min().item():.8f}")
    print(f"  Max: {spatial_sum.max().item():.8f}")
    print(f"  Mean: {spatial_sum.mean().item():.8f}")
    
    # Check for zero sums (would cause division issues)
    zero_sums = (spatial_sum == 0).sum().item()
    if zero_sums > 0:
        print(f"  ⚠️  WARNING: {zero_sums} tokens have zero spatial sum!")
    
    # Step 2: Normalize to create probability distribution
    prob = attn_map / (spatial_sum + eps)
    print(f"\nStep 2: Normalized probability")
    print(f"  Shape: {prob.shape}")
    print(f"  Min: {prob.min().item():.8f}")
    print(f"  Max: {prob.max().item():.8f}")
    print(f"  Mean: {prob.mean().item():.8f}")
    
    # Check if probabilities sum to 1 for each token
    prob_sums = prob.sum(dim=1)
    print(f"  Probability sums per token (should be ~1.0):")
    print(f"    Min: {prob_sums.min().item():.8f}")
    print(f"    Max: {prob_sums.max().item():.8f}")
    print(f"    Mean: {prob_sums.mean().item():.8f}")
    
    # Step 3: Calculate log(prob)
    log_prob = torch.log(prob + eps)
    print(f"\nStep 3: Log probability")
    print(f"  Min: {log_prob.min().item():.8f}")
    print(f"  Max: {log_prob.max().item():.8f}")
    print(f"  Mean: {log_prob.mean().item():.8f}")
    
    # Step 4: Calculate p * log(p)
    p_log_p = prob * log_prob
    print(f"\nStep 4: p * log(p)")
    print(f"  Min: {p_log_p.min().item():.8f}")
    print(f"  Max: {p_log_p.max().item():.8f}")
    print(f"  Mean: {p_log_p.mean().item():.8f}")
    print(f"  Sum: {p_log_p.sum().item():.8f}")
    
    # Step 5: Sum over spatial dimension
    entropy = p_log_p.sum(dim=1)
    print(f"\nStep 5: Entropy per token (sum over spatial)")
    print(f"  Shape: {entropy.shape}")
    print(f"  Min: {entropy.min().item():.8f}")
    print(f"  Max: {entropy.max().item():.8f}")
    print(f"  Mean: {entropy.mean().item():.8f}")
    
    # Step 6: Final mean
    final_entropy = entropy.mean()
    print(f"\nStep 6: Final entropy (mean over batch and tokens)")
    print(f"  Value: {final_entropy.item():.8f}")
    
    # Theoretical max entropy
    max_entropy_theoretical = -torch.log(torch.tensor(1.0 / s))
    print(f"\nTheoretical maximum entropy: {max_entropy_theoretical.item():.4f}")
    print(f"  (for uniform distribution over {s} spatial locations)")
    
    # Check if attention is too concentrated
    max_vals_per_token = prob.max(dim=1)[0]
    print(f"\nConcentration analysis:")
    print(f"  Max probability per token (higher = more concentrated):")
    print(f"    Min: {max_vals_per_token.min().item():.8f}")
    print(f"    Max: {max_vals_per_token.max().item():.8f}")
    print(f"    Mean: {max_vals_per_token.mean().item():.8f}")
    
    if max_vals_per_token.mean() > 0.9:
        print(f"  ⚠️  Attention is highly concentrated (mean max prob > 0.9)")
    
    return final_entropy


def create_test_cases():
    """Create test attention maps with known entropy values."""
    print("\n" + "="*60)
    print("TEST CASES")
    print("="*60)
    
    # Test 1: Uniform distribution (maximum entropy)
    print("\n--- Test 1: Uniform distribution ---")
    uniform = torch.ones(1, 16, 3) / 16  # uniform over spatial dim
    result = diagnose_max_entropy_loss(uniform, verbose=False)
    print(f"Result: {result.item():.8f}")
    print("Expected: Close to log(16) = 2.77")
    
    # Test 2: Concentrated attention (low entropy)
    print("\n--- Test 2: Concentrated attention ---")
    concentrated = torch.zeros(1, 16, 3)
    concentrated[0, 0, :] = 1.0  # All attention on first spatial location
    result = diagnose_max_entropy_loss(concentrated, verbose=False)
    print(f"Result: {result.item():.8f}")
    print("Expected: Close to 0 (minimum entropy)")
    
    # Test 3: All zeros (edge case)
    print("\n--- Test 3: All zeros ---")
    zeros = torch.zeros(1, 16, 3)
    result = diagnose_max_entropy_loss(zeros, verbose=False)
    print(f"Result: {result.item():.8f}")
    print("Expected: Should handle gracefully with eps")


def check_attention_extraction(controller, res, eps=1e-10):
    """
    Check if attention maps are being extracted correctly.
    """
    print(f"\n{'='*60}")
    print(f"ATTENTION EXTRACTION DIAGNOSIS")
    print(f"{'='*60}")
    
    attn_store = getattr(controller, 'step_store', getattr(controller, 'attention_store', {}))
    
    print(f"\nController type: {type(controller)}")
    print(f"Available keys in attention store: {list(attn_store.keys())}")
    
    for key in attn_store.keys():
        items = attn_store[key]
        print(f"\nKey: {key}")
        print(f"  Number of attention maps: {len(items)}")
        
        if len(items) > 0:
            for i, attn in enumerate(items[:3]):  # Show first 3
                print(f"  Map {i}: shape={attn.shape}, device={attn.device}, "
                      f"min={attn.min().item():.4f}, max={attn.max().item():.4f}, "
                      f"mean={attn.mean().item():.4f}")


if __name__ == "__main__":
    print("="*60)
    print("DIAGNOSTIC SCRIPT FOR ZERO ENTROPY ISSUE")
    print("="*60)
    
    # Run test cases
    create_test_cases()
    
    print("\n\nTo use this diagnostic in your code, add:")
    print("=" * 60)
    print("""
# After extracting attention map
from diagnostic_entropy import diagnose_max_entropy_loss, analyze_attention_map

# Before calculating entropy loss:
analyze_attention_map(after_true_label_attention_map, "True Label Attention")
neg_entropy = diagnose_max_entropy_loss(after_true_label_attention_map)
    """)
