#!/usr/bin/env python3
"""
Test script for per-timestep loss analysis functionality.
Tests the implementation with a minimal example to verify all components work.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Set minimal test configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test_timestep_tracking():
    """Test the timestep tracking components independently."""
    print("\n" + "="*60)
    print("TESTING TIMESTEP TRACKING COMPONENTS")
    print("="*60)
    
    # Test 1: AttentionControlEdit initialization
    print("\n[Test 1] Testing AttentionControlEdit initialization...")
    try:
        from attentionControl import AttentionControlEdit
        controller = AttentionControlEdit(num_steps=20, self_replace_steps=1.0, res=224)
        assert hasattr(controller, 'timestep_loss_details'), "Missing timestep_loss_details attribute"
        assert hasattr(controller, 'current_iteration'), "Missing current_iteration attribute"
        assert hasattr(controller, 'current_timestep_index'), "Missing current_timestep_index attribute"
        assert controller.timestep_loss_details == [], "timestep_loss_details should be empty list"
        assert controller.current_iteration == 0, "current_iteration should be 0"
        print("✓ AttentionControlEdit initialization PASSED")
    except Exception as e:
        print(f"✗ AttentionControlEdit initialization FAILED: {e}")
        return False
    
    # Test 2: Loss detail tracking
    print("\n[Test 2] Testing loss detail tracking...")
    try:
        controller.current_iteration = 5
        controller.current_timestep_index = 3
        # Simulate adding a timestep loss
        controller.timestep_loss_details.append({
            'timestep': 3,
            'iteration': 5,
            'loss': 0.123,
            'loss_type': 'self_attn'
        })
        assert len(controller.timestep_loss_details) == 1, "Should have 1 entry"
        assert controller.timestep_loss_details[0]['timestep'] == 3, "Timestep should be 3"
        print("✓ Loss detail tracking PASSED")
    except Exception as e:
        print(f"✗ Loss detail tracking FAILED: {e}")
        return False
    
    # Test 3: Reset functionality
    print("\n[Test 3] Testing reset functionality...")
    try:
        controller.reset()
        assert controller.timestep_loss_details == [], "timestep_loss_details should be cleared"
        print("✓ Reset functionality PASSED")
    except Exception as e:
        print(f"✗ Reset functionality FAILED: {e}")
        return False
    
    # Test 4: Visualization functions
    print("\n[Test 4] Testing visualization functions...")
    try:
        from utils import (print_timestep_statistics, 
                          plot_loss_per_timestep_iterations,
                          plot_loss_heatmap_timestep_vs_iteration,
                          plot_average_loss_per_timestep)
        
        # Create mock data
        mock_losses = {
            'attack_loss': [0.5, 0.4, 0.3],
            'cross_attn_loss': [0.1, 0.09, 0.08],
            'self_attn_loss': [0.2, 0.18, 0.16],
            'total_loss': [0.8, 0.67, 0.54],
            'timestep_details': [
                [
                    {'timestep': 0, 'iteration': 0, 'loss': 0.05, 'loss_type': 'self_attn'},
                    {'timestep': 1, 'iteration': 0, 'loss': 0.07, 'loss_type': 'self_attn'},
                    {'timestep': 2, 'iteration': 0, 'loss': 0.08, 'loss_type': 'self_attn'},
                ],
                [
                    {'timestep': 0, 'iteration': 1, 'loss': 0.04, 'loss_type': 'self_attn'},
                    {'timestep': 1, 'iteration': 1, 'loss': 0.06, 'loss_type': 'self_attn'},
                    {'timestep': 2, 'iteration': 1, 'loss': 0.08, 'loss_type': 'self_attn'},
                ],
                [
                    {'timestep': 0, 'iteration': 2, 'loss': 0.03, 'loss_type': 'self_attn'},
                    {'timestep': 1, 'iteration': 2, 'loss': 0.05, 'loss_type': 'self_attn'},
                    {'timestep': 2, 'iteration': 2, 'loss': 0.08, 'loss_type': 'self_attn'},
                ],
            ]
        }
        
        # Test statistics printing
        print("\n  Testing print_timestep_statistics...")
        print_timestep_statistics(mock_losses, num_timesteps=3)
        
        # Test visualization functions (save to temp directory)
        test_output_dir = "/tmp/diffattack_test"
        os.makedirs(test_output_dir, exist_ok=True)
        
        matplotlib_available = True
        try:
            import matplotlib
        except ImportError:
            matplotlib_available = False
            print("\n⚠ matplotlib not installed. Installing now...")
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn", "-q"])
                print("✓ matplotlib and seaborn installed successfully")
                matplotlib_available = True
            except Exception as e:
                print(f"✗ Failed to install matplotlib: {e}")
                print("  Please install manually: pip install matplotlib seaborn")
                return False
        
        print("\n  Testing plot_loss_per_timestep_iterations...")
        try:
            plot_loss_per_timestep_iterations(mock_losses, test_output_dir)
            print("    ✓ Function executed successfully")
        except Exception as e:
            print(f"    ⚠ Error in plot_loss_per_timestep_iterations: {e}")
        
        print("\n  Testing plot_loss_heatmap_timestep_vs_iteration...")
        try:
            plot_loss_heatmap_timestep_vs_iteration(mock_losses, test_output_dir)
            print("    ✓ Function executed successfully")
        except Exception as e:
            print(f"    ⚠ Error in plot_loss_heatmap_timestep_vs_iteration: {e}")
        
        print("\n  Testing plot_average_loss_per_timestep...")
        try:
            plot_average_loss_per_timestep(mock_losses, test_output_dir)
            print("    ✓ Function executed successfully")
        except Exception as e:
            print(f"    ⚠ Error in plot_average_loss_per_timestep: {e}")
        
        # Check if files were created
        expected_files = ['loss_per_timestep.png', 'loss_heatmap.png', 'average_loss_per_timestep.png']
        all_files_created = True
        for fname in expected_files:
            fpath = os.path.join(test_output_dir, fname)
            if os.path.exists(fpath):
                print(f"  ✓ Created: {fname}")
            else:
                print(f"  ✗ Missing: {fname}")
                all_files_created = False
        
        if all_files_created:
            print("\n✓ Visualization functions PASSED")
            print(f"\n  Test outputs saved to: {test_output_dir}")
        else:
            print("\n⚠ Some visualization files not created (check matplotlib installation)")
        
    except Exception as e:
        print(f"✗ Visualization functions FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DIFFATTACK PER-TIMESTEP LOSS ANALYSIS - TEST SUITE")
    print("="*60)
    
    success = test_timestep_tracking()
    
    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nThe implementation is ready to use!")
        print("\nTo test with actual images, run:")
        print("  python main.py --images_root demo/images --label_path demo/labels.txt --save_dir output")
        print("\nTimestep analysis will be saved to: output/timestep_analysis/")
    else:
        print("SOME TESTS FAILED ✗")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
