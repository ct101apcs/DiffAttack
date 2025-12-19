import math
from typing import Iterable, Sequence, Tuple
import numpy as np
import torch
from PIL import Image


def view_images(images: np.ndarray,
                num_rows: int = 1,
                offset: int = 0,
                show: bool = False,
                save_path: str = None) -> None:
    """Save or show a simple grid of images.

    Expects images as N x H x W x C in [0,255] (uint8 or float).
    Creates a single-row or multi-row grid and optionally saves it.
    """
    if images is None:
        return

    imgs = images
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()

    if imgs.ndim == 3:
        # Single image HWC -> add N=1
        imgs = imgs[None]
    assert imgs.ndim == 4, f"Expected images as [N,H,W,C], got shape {imgs.shape}"

    N, H, W, C = imgs.shape
    if imgs.dtype != np.uint8:
        imgs = np.clip(imgs, 0, 255).astype(np.uint8)

    if num_rows <= 0:
        num_rows = 1
    num_cols = math.ceil(N / num_rows)

    grid = np.zeros((num_rows * H, num_cols * W, C), dtype=np.uint8)

    for idx in range(N):
        r = idx // num_cols
        c = idx % num_cols
        grid[r*H:(r+1)*H, c*W:(c+1)*W, :] = imgs[idx]

    img = Image.fromarray(grid)
    if save_path:
        img.save(save_path)
    if show:
        img.show()


@torch.no_grad()
def aggregate_attention(prompt: Sequence[str],
                        controller,
                        res: int,
                        from_where: Tuple[str, ...] = ("up", "down"),
                        is_cross: bool = True,
                        select: int = 0,
                        is_cpu: bool = False) -> torch.Tensor:
    """Aggregate attention maps from controller into a [res, res, token] tensor.

    - res: target spatial resolution for the attention map (e.g., args.res // 32)
    - from_where: which UNet blocks to use ("up", "down", "mid")
    - is_cross: choose cross/self attention tensors
    - select: sample index to select from the controller's batch (0 or 1)
    - returns: tensor of shape [res, res, token_len]
    """
    # Prefer averaged attention over steps when available
    attn_store = controller.get_average_attention() if hasattr(controller, 'get_average_attention') else getattr(controller, 'attention_store', {})

    keys = [f"{w}_{'cross' if is_cross else 'self'}" for w in from_where]
    maps = []
    bsz = getattr(controller, 'batch_size', 2)

    for key in keys:
        for attn in attn_store.get(key, []):
            # attn: [batch*heads, query, key]
            if not isinstance(attn, torch.Tensor):
                continue
            bh, q, k = attn.shape
            h = max(1, bh // max(1, bsz))
            try:
                attn = attn.reshape(bsz, h, q, k).mean(1)  # [bsz, q, k]
            except Exception:
                # If reshape fails, treat whole dim as heads=1
                attn = attn.reshape(bh, q, k).mean(0, keepdim=True)
                bsz = 1

            # Try to reshape query tokens into spatial map [res, res]
            if q == res * res:
                att_sel = attn[select]  # [q, k]
                att_sel = att_sel.reshape(res, res, k)  # [res, res, k]
                maps.append(att_sel)
            else:
                # Attempt square reshape via sqrt; fallback: skip
                sq = int(math.sqrt(q))
                if sq * sq == q:
                    att_sel = attn[select].reshape(sq, sq, k)
                    # Resize to [res,res] via bilinear on channels as batch
                    att_chw = att_sel.permute(2, 0, 1).unsqueeze(0)  # [1,k,sq,sq]
                    att_resized = torch.nn.functional.interpolate(att_chw, size=(res, res), mode='bilinear', align_corners=False)
                    att_resized = att_resized.squeeze(0).permute(1, 2, 0)  # [res,res,k]
                    maps.append(att_resized)
                else:
                    # Cannot map spatially; skip this layer
                    continue

    if not maps:
        # Fallback: return zeros with a reasonable token dim guess (77)
        device = 'cpu' if is_cpu else (maps[0].device if maps else 'cpu')
        return torch.zeros(res, res, 77, device=device)

    att = torch.stack(maps, dim=0).mean(0)  # [res,res,k]
    return att.cpu() if is_cpu else att


def print_timestep_statistics(all_iteration_losses, num_timesteps):
    """Print per-timestep loss statistics to console.
    
    Args:
        all_iteration_losses: Dict containing 'timestep_details' list
        num_timesteps: Total number of timesteps
    """
    print("\n" + "="*50)
    print("PER-TIMESTEP LOSS STATISTICS")
    print("="*50)
    
    # Aggregate self-attention losses per timestep across all iterations
    timestep_losses = {}
    for details_list in all_iteration_losses['timestep_details']:
        for detail in details_list:
            ts = detail['timestep']
            if ts not in timestep_losses:
                timestep_losses[ts] = []
            timestep_losses[ts].append(detail['loss'])
    
    # Compute statistics
    total_loss_sum = 0
    stats = []
    for ts in sorted(timestep_losses.keys()):
        losses = timestep_losses[ts]
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        total_loss_sum += avg_loss
        stats.append((ts, avg_loss, std_loss))
    
    # Print with contribution percentages
    for ts, avg_loss, std_loss in stats:
        contribution = (avg_loss / total_loss_sum * 100) if total_loss_sum > 0 else 0
        print(f"Timestep {ts:2d}: Avg Loss={avg_loss:.6f}, Std={std_loss:.6f}, Contribution={contribution:.1f}%")
    
    print(f"\nTotal timesteps: {num_timesteps}")
    uniform_weight = 100.0 / num_timesteps if num_timesteps > 0 else 0
    print(f"Uniform weight would be: {uniform_weight:.2f}%")
    
    if stats:
        max_contrib = max(stats, key=lambda x: x[1])
        min_contrib = min(stats, key=lambda x: x[1])
        max_pct = (max_contrib[1] / total_loss_sum * 100) if total_loss_sum > 0 else 0
        min_pct = (min_contrib[1] / total_loss_sum * 100) if total_loss_sum > 0 else 0
        print(f"Max contribution: {max_pct:.1f}% (timestep {max_contrib[0]})")
        print(f"Min contribution: {min_pct:.1f}% (timestep {min_contrib[0]})")
    print("="*50 + "\n")


def plot_loss_per_timestep_iterations(all_iteration_losses, output_dir):
    """Plot loss curves per timestep for key iterations.
    
    Creates a 2x2 grid showing self-attention, cross-attention, attack, and total losses
    across timesteps for selected iterations.
    
    Args:
        all_iteration_losses: Dict with keys 'timestep_details', 'attack_loss', etc.
        output_dir: Directory to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Select key iterations to plot
        num_iterations = len(all_iteration_losses['total_loss'])
        if num_iterations >= 30:
            key_iterations = [0, 10, 20, 29]
        elif num_iterations >= 20:
            key_iterations = [0, num_iterations//3, 2*num_iterations//3, num_iterations-1]
        elif num_iterations >= 10:
            key_iterations = [0, num_iterations//2, num_iterations-1]
        else:
            key_iterations = [0, num_iterations-1]
        
        # Aggregate losses per timestep for each iteration
        def get_timestep_losses_by_iteration(timestep_details):
            iter_timestep_losses = {}
            for iter_idx, details_list in enumerate(timestep_details):
                timestep_losses = {}
                for detail in details_list:
                    ts = detail['timestep']
                    if ts not in timestep_losses:
                        timestep_losses[ts] = []
                    timestep_losses[ts].append(detail['loss'])
                # Average losses at same timestep (multiple attention layers)
                iter_timestep_losses[iter_idx] = {ts: np.mean(losses) for ts, losses in timestep_losses.items()}
            return iter_timestep_losses
        
        self_attn_by_iter = get_timestep_losses_by_iteration(all_iteration_losses['timestep_details'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Loss per Timestep at Key Iterations', fontsize=16, fontweight='bold')
        
        # Plot 1: Self-attention loss
        ax = axes[0, 0]
        for iter_idx in key_iterations:
            if iter_idx in self_attn_by_iter:
                timesteps = sorted(self_attn_by_iter[iter_idx].keys())
                losses = [self_attn_by_iter[iter_idx][ts] for ts in timesteps]
                ax.plot(timesteps, losses, marker='o', label=f'Iter {iter_idx}', linewidth=2)
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Self-Attention Loss', fontsize=11)
        ax.set_title('Self-Attention Loss per Timestep', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cross-attention loss (use total cross_attn_loss / num_timesteps as approximation)
        ax = axes[0, 1]
        if self_attn_by_iter:
            num_timesteps = len(list(self_attn_by_iter[0].keys())) if 0 in self_attn_by_iter else 1
            for iter_idx in key_iterations:
                if iter_idx < len(all_iteration_losses['cross_attn_loss']):
                    timesteps = sorted(self_attn_by_iter[iter_idx].keys()) if iter_idx in self_attn_by_iter else range(num_timesteps)
                    # Approximate uniform distribution
                    avg_cross = all_iteration_losses['cross_attn_loss'][iter_idx] / max(num_timesteps, 1)
                    losses = [avg_cross] * len(timesteps)
                    ax.plot(timesteps, losses, marker='s', label=f'Iter {iter_idx}', linewidth=2, linestyle='--')
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Cross-Attention Variance (avg)', fontsize=11)
        ax.set_title('Cross-Attention Variance per Timestep', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Attack loss (approximated per timestep)
        ax = axes[1, 0]
        if self_attn_by_iter:
            num_timesteps = len(list(self_attn_by_iter[0].keys())) if 0 in self_attn_by_iter else 1
            for iter_idx in key_iterations:
                if iter_idx < len(all_iteration_losses['attack_loss']):
                    timesteps = sorted(self_attn_by_iter[iter_idx].keys()) if iter_idx in self_attn_by_iter else range(num_timesteps)
                    avg_attack = all_iteration_losses['attack_loss'][iter_idx] / max(num_timesteps, 1)
                    losses = [avg_attack] * len(timesteps)
                    ax.plot(timesteps, losses, marker='^', label=f'Iter {iter_idx}', linewidth=2, linestyle=':')
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Attack Loss (avg)', fontsize=11)
        ax.set_title('Attack Loss per Timestep', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Total loss
        ax = axes[1, 1]
        for iter_idx in key_iterations:
            if iter_idx < len(all_iteration_losses['total_loss']):
                if iter_idx in self_attn_by_iter:
                    timesteps = sorted(self_attn_by_iter[iter_idx].keys())
                    total = all_iteration_losses['total_loss'][iter_idx]
                    losses = [total / max(len(timesteps), 1)] * len(timesteps)
                    ax.plot(timesteps, losses, marker='d', label=f'Iter {iter_idx}', linewidth=2, linestyle='-.')
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Total Loss (avg)', fontsize=11)
        ax.set_title('Total Loss per Timestep', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'loss_per_timestep.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")
        
    except Exception as e:
        print(f"⚠ Error in plot_loss_per_timestep_iterations: {e}")


def plot_loss_heatmap_timestep_vs_iteration(all_iteration_losses, output_dir):
    """Plot heatmap of self-attention loss: timestep (Y) vs iteration (X).
    
    Args:
        all_iteration_losses: Dict containing 'timestep_details'
        output_dir: Directory to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Build matrix: rows=timesteps, cols=iterations
        timestep_details = all_iteration_losses['timestep_details']
        num_iterations = len(timestep_details)
        
        # Find max timestep
        max_timestep = 0
        for details_list in timestep_details:
            for detail in details_list:
                max_timestep = max(max_timestep, detail['timestep'])
        
        num_timesteps = max_timestep + 1
        loss_matrix = np.zeros((num_timesteps, num_iterations))
        
        # Aggregate losses
        for iter_idx, details_list in enumerate(timestep_details):
            timestep_losses = {}
            for detail in details_list:
                ts = detail['timestep']
                if ts not in timestep_losses:
                    timestep_losses[ts] = []
                timestep_losses[ts].append(detail['loss'])
            
            for ts, losses in timestep_losses.items():
                loss_matrix[ts, iter_idx] = np.mean(losses)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(loss_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Timestep', fontsize=12)
        ax.set_title('Self-Attention Loss Heatmap: Timestep vs Iteration', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loss Magnitude', fontsize=11)
        
        # Set ticks
        ax.set_xticks(np.arange(0, num_iterations, max(1, num_iterations // 10)))
        ax.set_yticks(np.arange(0, num_timesteps, max(1, num_timesteps // 10)))
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'loss_heatmap.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")
        
    except Exception as e:
        print(f"⚠ Error in plot_loss_heatmap_timestep_vs_iteration: {e}")


def plot_average_loss_per_timestep(all_iteration_losses, output_dir):
    """Plot average loss per timestep with error bars.
    
    Shows which timesteps contribute most on average across all iterations.
    
    Args:
        all_iteration_losses: Dict containing 'timestep_details'
        output_dir: Directory to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Aggregate losses per timestep across all iterations
        timestep_losses = {}
        for details_list in all_iteration_losses['timestep_details']:
            for detail in details_list:
                ts = detail['timestep']
                if ts not in timestep_losses:
                    timestep_losses[ts] = []
                timestep_losses[ts].append(detail['loss'])
        
        # Compute mean and std
        timesteps = sorted(timestep_losses.keys())
        means = [np.mean(timestep_losses[ts]) for ts in timesteps]
        stds = [np.std(timestep_losses[ts]) for ts in timesteps]
        
        # Compute contribution percentages
        total_loss = sum(means)
        contributions = [(m / total_loss * 100) if total_loss > 0 else 0 for m in means]
        uniform_baseline = 100.0 / len(timesteps) if len(timesteps) > 0 else 0
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average loss with error bars
        ax = axes[0]
        bars = ax.bar(timesteps, means, yerr=stds, capsize=5, alpha=0.7, 
                     color='steelblue', edgecolor='navy', linewidth=1.5)
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Average Self-Attention Loss', fontsize=12)
        ax.set_title('Average Loss per Timestep (± Std)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Contribution percentage
        ax = axes[1]
        bars = ax.bar(timesteps, contributions, alpha=0.7, 
                     color='coral', edgecolor='darkred', linewidth=1.5)
        ax.axhline(y=uniform_baseline, color='black', linestyle='--', linewidth=2, 
                  label=f'Uniform baseline ({uniform_baseline:.1f}%)')
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Contribution (%)', fontsize=12)
        ax.set_title('Loss Contribution per Timestep', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'average_loss_per_timestep.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")
        
    except Exception as e:
        print(f"⚠ Error in plot_average_loss_per_timestep: {e}")


def plot_learned_vs_initial_weights(weight_network, output_dir, weight_history=None):
    """
    Plot comparison of initial vs learned timestep weights.
    
    Creates a multi-panel visualization showing:
    1. Bar comparison of initial vs learned weights (self-attention)
    2. Bar comparison of initial vs learned weights (cross-attention)
    3. Weight evolution during training (if history provided)
    4. Summary statistics
    
    Args:
        weight_network: TimestepWeightNetwork instance with learned weights
        output_dir: Directory to save the plot
        weight_history: Optional list of weight snapshots during training
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Get weight comparison data
        comparison = weight_network.get_weight_comparison()
        num_timesteps = comparison['num_timesteps']
        timesteps = list(range(num_timesteps))
        
        # Calculate figure layout based on available data
        has_history = weight_history is not None and len(weight_history) > 0
        
        if has_history:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax1, ax2 = axes
            ax3, ax4 = None, None
        
        bar_width = 0.35
        x = np.array(timesteps)
        
        # ===== Plot 1: Self-Attention Weights =====
        init_self = comparison['initial_self_weights']
        learned_self = comparison['learned_self_weights']
        
        bars1 = ax1.bar(x - bar_width/2, init_self, bar_width, 
                       label='Initial (Increasing Schedule)', 
                       color='lightcoral', edgecolor='darkred', alpha=0.8)
        bars2 = ax1.bar(x + bar_width/2, learned_self, bar_width,
                       label='Learned', 
                       color='steelblue', edgecolor='navy', alpha=0.8)
        
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Weight', fontsize=12)
        ax1.set_title('Self-Attention Weights: Initial vs Learned', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars1, init_self):
            ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        for bar, val in zip(bars2, learned_self):
            ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        
        # ===== Plot 2: Cross-Attention Weights =====
        init_cross = comparison['initial_cross_weights']
        learned_cross = comparison['learned_cross_weights']
        
        bars3 = ax2.bar(x - bar_width/2, init_cross, bar_width,
                       label='Initial (Increasing Schedule)',
                       color='lightgreen', edgecolor='darkgreen', alpha=0.8)
        bars4 = ax2.bar(x + bar_width/2, learned_cross, bar_width,
                       label='Learned',
                       color='mediumpurple', edgecolor='indigo', alpha=0.8)
        
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Weight', fontsize=12)
        ax2.set_title('Cross-Attention Weights: Initial vs Learned', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars3, init_cross):
            ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        for bar, val in zip(bars4, learned_cross):
            ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        
        # ===== Plot 3: Weight Evolution (if history available) =====
        if has_history and ax3 is not None:
            iterations = [h['iteration'] for h in weight_history]
            
            # Plot self-attention weight evolution
            for ts in timesteps:
                weights = [h['self_weights'][ts] for h in weight_history]
                ax3.plot(iterations, weights, marker='o', markersize=2, 
                        label=f'Timestep {ts}', linewidth=1.5)
            
            ax3.set_xlabel('Iteration', fontsize=12)
            ax3.set_ylabel('Self-Attention Weight', fontsize=12)
            ax3.set_title('Self-Attention Weight Evolution During Training', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=9, loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # ===== Plot 4: Statistics Summary =====
        if ax4 is not None:
            ax4.axis('off')
            
            stats_text = [
                "LEARNED WEIGHTS STATISTICS",
                "=" * 40,
                "",
                "Self-Attention:",
                f"  Max change timestep: {comparison['self_max_change_timestep']}",
                f"  Total weight shift: {comparison['self_total_shift']:.4f}",
                f"  Weight change: {comparison['self_weight_change']}",
                "",
                "Cross-Attention:",
                f"  Max change timestep: {comparison['cross_max_change_timestep']}",
                f"  Total weight shift: {comparison['cross_total_shift']:.4f}",
                f"  Weight change: {comparison['cross_weight_change']}",
                "",
                "Initial Schedule: [0.05, 0.08, 0.13, 0.24, 0.50]",
                "(Based on empirical analysis showing late",
                "timesteps contribute more to loss)"
            ]
            
            ax4.text(0.1, 0.9, '\n'.join(stats_text), transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Learnable Timestep Weights Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'learned_weights_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")
        
        # Print statistics to console
        print("\n" + "="*50)
        print("LEARNED vs INITIAL WEIGHTS COMPARISON")
        print("="*50)
        print(f"\nSelf-Attention Weights:")
        print(f"  Initial:  {init_self}")
        print(f"  Learned:  {learned_self}")
        print(f"  Max change at timestep {comparison['self_max_change_timestep']}")
        print(f"  Total weight shift: {comparison['self_total_shift']:.4f}")
        
        print(f"\nCross-Attention Weights:")
        print(f"  Initial:  {init_cross}")
        print(f"  Learned:  {learned_cross}")
        print(f"  Max change at timestep {comparison['cross_max_change_timestep']}")
        print(f"  Total weight shift: {comparison['cross_total_shift']:.4f}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"⚠ Error in plot_learned_vs_initial_weights: {e}")
        import traceback
        traceback.print_exc()
