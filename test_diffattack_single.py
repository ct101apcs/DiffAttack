#!/usr/bin/env python3
"""
Test script to verify DiffAttack works with a single image
"""
import torch
import numpy as np
from PIL import Image
import os
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attentionControl import AttentionControlEdit
import diff_latent_attack

# Simple test with one image
def test_single_diffusion_attack():
    # Setup arguments (minimal)
    class Args:
        def __init__(self):
            self.dataset_name = "imagenet_compatible"
            self.diffusion_steps = 20
            self.guidance = 2.5
            self.res = 224
            self.start_step = 15
            self.iterations = 30
            self.pretrained_diffusion_path = "Manojb/stable-diffusion-2-base"
            # Add missing attributes from main.py argument parser
            self.is_apply_mask = False
            self.is_hard_mask = False
            self.eps = 16.0/255.0
            self.num_iter = 10
            self.momentum = 1.0
            self.attack_loss_weight = 1.0
            self.cross_attn_loss_weight = 1.0
            self.self_attn_loss_weight = 1.0
    
    args = Args()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Load one test image
    img_path = "demo/images/demo_01.png"
    if not os.path.exists(img_path):
        print(f"❌ Test image not found: {img_path}")
        return
    
    image = Image.open(img_path).convert('RGB')
    print(f"✅ Loaded image: {img_path}")
    
    # Test label (pretending it's class 281 - tabby cat)
    label = np.array([281])  # Single element numpy array
    
    try:
        # Setup diffusion pipeline 
        print("🚀 Loading Stable Diffusion pipeline...")
        ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path).to('cuda:0')  # GPU 7 mapped to cuda:0
        ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
        print("✅ Pipeline loaded successfully")
        
        # Create controller
        self_replace_steps = 1.0
        controller = AttentionControlEdit(args.diffusion_steps, self_replace_steps, args.res)
        print("✅ Controller created successfully")
        
        # Run DiffAttack
        print("🎯 Running DiffAttack...")
        save_path = "test_single_output"
        os.makedirs(save_path, exist_ok=True)
        
        adv_image, clean_acc, adv_acc = diff_latent_attack.diffattack(
            ldm_stable, label, controller,
            num_inference_steps=args.diffusion_steps,
            guidance_scale=args.guidance,
            image=image,
            save_path=save_path, 
            res=args.res, 
            model_name="resnet",  # Use correct model name from other_attacks.py
            start_step=args.start_step,
            iterations=args.iterations, 
            args=args
        )
        
        print(f"✅ DiffAttack completed!")
        print(f"   Clean accuracy: {clean_acc:.3f}")
        print(f"   Adversarial accuracy: {adv_acc:.3f}")
        print(f"   Attack success: {'Yes' if adv_acc < clean_acc else 'No'}")
        
    except Exception as e:
        print(f"❌ Error in DiffAttack: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_diffusion_attack()