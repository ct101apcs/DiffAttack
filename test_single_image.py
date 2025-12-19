#!/usr/bin/env python3
"""
Quick integration test with a single demo image to verify timestep analysis.
"""

import os
import sys
import argparse

# Setup minimal arguments for testing
sys.argv = [
    'test_single_image.py',
    '--save_dir', 'test_output_single',
    '--images_root', 'demo/images',
    '--label_path', 'demo/labels.txt',
    '--diffusion_steps', '10',  # Reduced for faster testing
    '--start_step', '7',
    '--iterations', '5',  # Reduced for faster testing
    '--res', '224',
    '--model_name', 'resnet',
    '--guidance', '2.5',
    '--attack_loss_weight', '10',
    '--cross_attn_loss_weight', '10000',
    '--self_attn_loss_weight', '100'
]

print("\n" + "="*70)
print("TESTING PER-TIMESTEP LOSS ANALYSIS WITH SINGLE DEMO IMAGE")
print("="*70)
print("\nConfiguration:")
print("  - Dataset: demo/images (first image only)")
print("  - Diffusion steps: 10 (reduced for testing)")
print("  - Attack iterations: 5 (reduced for testing)")
print("  - Output: test_output_single/")
print("\nThis will:")
print("  1. Run attack on first demo image")
print("  2. Track losses per timestep")
print("  3. Generate timestep analysis visualizations")
print("="*70 + "\n")

# Import and run main
import main

# Note: main.py will process only the first image from demo/images
# The timestep analysis plots will be saved to:
# - test_output_single/timestep_analysis/loss_per_timestep.png
# - test_output_single/timestep_analysis/loss_heatmap.png
# - test_output_single/timestep_analysis/average_loss_per_timestep.png
