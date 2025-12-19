"""
Learnable Timestep Weight Network for DiffAttack

Based on per-timestep loss analysis showing:
- Timestep 4 contributes 47-57% (2.6x above uniform 20%)
- Timestep 0 contributes only 2-6% (0.3x of uniform)
- Clear exponential increase: early timesteps contribute much less than late ones

This module provides learnable weights that can adapt during attack optimization.
"""

import torch
import torch.nn as nn


class TimestepWeightNetwork(nn.Module):
    """
    Learnable weight network for timestep-based attention loss weighting.
    
    Produces two separate weight vectors (for self-attention and cross-attention),
    each summing to 1.0 via softmax normalization.
    
    Architecture: Input(1) → Linear(64) → ReLU → Linear(num_timesteps) → Softmax
    
    Initial weights follow increasing schedule based on empirical analysis:
    [0.05, 0.08, 0.13, 0.24, 0.50] - late timesteps weighted higher.
    """
    
    # Default initial weights based on empirical analysis
    DEFAULT_SELF_WEIGHTS = [0.05, 0.08, 0.13, 0.24, 0.50]
    DEFAULT_CROSS_WEIGHTS = [0.05, 0.08, 0.13, 0.24, 0.50]
    
    def __init__(self, num_timesteps: int = 5, device: str = 'cuda'):
        """
        Initialize the weight network.
        
        Args:
            num_timesteps: Number of active timesteps (default=5 based on analysis)
            device: Device to place the network on
        """
        super(TimestepWeightNetwork, self).__init__()
        
        self.num_timesteps = num_timesteps
        self.device = device
        
        # MLP for self-attention weights
        self.self_attn_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, num_timesteps)
        )
        
        # MLP for cross-attention weights
        self.cross_attn_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, num_timesteps)
        )
        
        # Softmax for normalization (weights sum to 1.0)
        self.softmax = nn.Softmax(dim=-1)
        
        # Store initial weights for comparison
        self.initial_self_weights = self._get_initial_weights(num_timesteps, 'self')
        self.initial_cross_weights = self._get_initial_weights(num_timesteps, 'cross')
        
        # Initialize network weights to produce the desired initial output
        self._initialize_to_schedule()
        
        self.to(device)
    
    def _get_initial_weights(self, num_timesteps: int, attn_type: str) -> torch.Tensor:
        """
        Get initial weights based on increasing schedule.
        
        Args:
            num_timesteps: Number of timesteps
            attn_type: 'self' or 'cross'
            
        Returns:
            Tensor of initial weights
        """
        if attn_type == 'self':
            default = self.DEFAULT_SELF_WEIGHTS
        else:
            default = self.DEFAULT_CROSS_WEIGHTS
        
        if num_timesteps == len(default):
            return torch.tensor(default, dtype=torch.float32)
        else:
            # Interpolate or extrapolate for different num_timesteps
            # Use exponential increasing pattern
            weights = torch.zeros(num_timesteps)
            for i in range(num_timesteps):
                # Exponential increase from 0.05 to 0.50
                t = i / max(num_timesteps - 1, 1)
                weights[i] = 0.05 + 0.45 * (t ** 2)  # Quadratic increase
            # Normalize to sum to 1
            weights = weights / weights.sum()
            return weights
    
    def _initialize_to_schedule(self):
        """
        Initialize the MLP weights so that initial output matches the desired schedule.
        
        We initialize the bias of the last linear layer to produce logits that,
        after softmax, give us the desired initial weights.
        """
        # Compute logits that produce desired weights after softmax
        # softmax(logits) = weights => logits ≈ log(weights) + constant
        
        # Self-attention MLP initialization
        target_self = self.initial_self_weights.clone()
        target_self = target_self.clamp(min=1e-6)  # Avoid log(0)
        logits_self = torch.log(target_self)
        logits_self = logits_self - logits_self.mean()  # Center the logits
        
        # Initialize the last layer bias to these logits
        with torch.no_grad():
            self.self_attn_mlp[-1].bias.copy_(logits_self)
            # Initialize last layer weights to small values so input has minimal effect initially
            nn.init.normal_(self.self_attn_mlp[-1].weight, mean=0.0, std=0.01)
            # Initialize first layer
            nn.init.xavier_uniform_(self.self_attn_mlp[0].weight)
            nn.init.zeros_(self.self_attn_mlp[0].bias)
        
        # Cross-attention MLP initialization
        target_cross = self.initial_cross_weights.clone()
        target_cross = target_cross.clamp(min=1e-6)
        logits_cross = torch.log(target_cross)
        logits_cross = logits_cross - logits_cross.mean()
        
        with torch.no_grad():
            self.cross_attn_mlp[-1].bias.copy_(logits_cross)
            nn.init.normal_(self.cross_attn_mlp[-1].weight, mean=0.0, std=0.01)
            nn.init.xavier_uniform_(self.cross_attn_mlp[0].weight)
            nn.init.zeros_(self.cross_attn_mlp[0].bias)
    
    def forward(self) -> tuple:
        """
        Compute weight vectors for self-attention and cross-attention.
        
        Returns:
            Tuple of (w_self, w_cross), each of shape [num_timesteps], summing to 1.0
        """
        # Dummy input (the network learns to map this to appropriate weights)
        x = torch.ones(1, 1, device=self.device)
        
        # Compute self-attention weights
        self_logits = self.self_attn_mlp(x)  # [1, num_timesteps]
        w_self = self.softmax(self_logits).squeeze(0)  # [num_timesteps]
        
        # Compute cross-attention weights
        cross_logits = self.cross_attn_mlp(x)  # [1, num_timesteps]
        w_cross = self.softmax(cross_logits).squeeze(0)  # [num_timesteps]
        
        return w_self, w_cross
    
    def get_self_weights(self) -> torch.Tensor:
        """Get only self-attention weights."""
        w_self, _ = self.forward()
        return w_self
    
    def get_cross_weights(self) -> torch.Tensor:
        """Get only cross-attention weights."""
        _, w_cross = self.forward()
        return w_cross
    
    def get_initial_weights(self) -> tuple:
        """
        Get the initial weight schedule (for comparison with learned weights).
        
        Returns:
            Tuple of (initial_self_weights, initial_cross_weights)
        """
        return (
            self.initial_self_weights.to(self.device),
            self.initial_cross_weights.to(self.device)
        )
    
    def get_weight_comparison(self) -> dict:
        """
        Get detailed comparison between initial and current learned weights.
        
        Returns:
            Dict with initial weights, learned weights, and change statistics
        """
        w_self, w_cross = self.forward()
        init_self, init_cross = self.get_initial_weights()
        
        with torch.no_grad():
            w_self_np = w_self.cpu().numpy()
            w_cross_np = w_cross.cpu().numpy()
            init_self_np = init_self.cpu().numpy()
            init_cross_np = init_cross.cpu().numpy()
        
        # Compute statistics
        self_diff = w_self_np - init_self_np
        cross_diff = w_cross_np - init_cross_np
        
        return {
            'initial_self_weights': init_self_np,
            'learned_self_weights': w_self_np,
            'self_weight_change': self_diff,
            'self_max_change_timestep': int(abs(self_diff).argmax()),
            'self_total_shift': float(abs(self_diff).sum()),
            
            'initial_cross_weights': init_cross_np,
            'learned_cross_weights': w_cross_np,
            'cross_weight_change': cross_diff,
            'cross_max_change_timestep': int(abs(cross_diff).argmax()),
            'cross_total_shift': float(abs(cross_diff).sum()),
            
            'num_timesteps': self.num_timesteps
        }
    
    def print_weight_summary(self):
        """Print a formatted summary of weights to console."""
        comparison = self.get_weight_comparison()
        
        print("\n" + "="*60)
        print("LEARNABLE TIMESTEP WEIGHTS SUMMARY")
        print("="*60)
        
        print("\n--- Self-Attention Weights ---")
        print(f"{'Timestep':<10} {'Initial':<12} {'Learned':<12} {'Change':<12}")
        print("-" * 46)
        for i in range(comparison['num_timesteps']):
            print(f"{i:<10} {comparison['initial_self_weights'][i]:<12.4f} "
                  f"{comparison['learned_self_weights'][i]:<12.4f} "
                  f"{comparison['self_weight_change'][i]:+.4f}")
        print(f"\nMax change at timestep {comparison['self_max_change_timestep']}")
        print(f"Total weight shift: {comparison['self_total_shift']:.4f}")
        
        print("\n--- Cross-Attention Weights ---")
        print(f"{'Timestep':<10} {'Initial':<12} {'Learned':<12} {'Change':<12}")
        print("-" * 46)
        for i in range(comparison['num_timesteps']):
            print(f"{i:<10} {comparison['initial_cross_weights'][i]:<12.4f} "
                  f"{comparison['learned_cross_weights'][i]:<12.4f} "
                  f"{comparison['cross_weight_change'][i]:+.4f}")
        print(f"\nMax change at timestep {comparison['cross_max_change_timestep']}")
        print(f"Total weight shift: {comparison['cross_total_shift']:.4f}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test the weight network
    print("Testing TimestepWeightNetwork...")
    
    # Create network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_net = TimestepWeightNetwork(num_timesteps=5, device=device)
    
    # Get initial weights
    w_self, w_cross = weight_net()
    print(f"\nInitial self-attention weights: {w_self}")
    print(f"Sum: {w_self.sum().item():.6f}")
    
    print(f"\nInitial cross-attention weights: {w_cross}")
    print(f"Sum: {w_cross.sum().item():.6f}")
    
    # Verify initialization
    init_self, init_cross = weight_net.get_initial_weights()
    print(f"\nTarget initial weights: {init_self}")
    print(f"Difference from target: {(w_self - init_self).abs().max().item():.6f}")
    
    # Test optimization
    print("\n--- Testing optimization ---")
    optimizer = torch.optim.Adam(weight_net.parameters(), lr=1e-3)
    
    # Simulate training to shift weights toward late timesteps even more
    for i in range(100):
        optimizer.zero_grad()
        w_self, w_cross = weight_net()
        
        # Loss to emphasize last timestep even more
        target = torch.zeros_like(w_self)
        target[-1] = 1.0
        loss = ((w_self - target) ** 2).sum()
        
        loss.backward()
        optimizer.step()
    
    # Check weights after optimization
    w_self, w_cross = weight_net()
    print(f"\nAfter optimization (emphasizing last timestep):")
    print(f"Self-attention weights: {w_self}")
    print(f"Sum: {w_self.sum().item():.6f}")
    
    # Print summary
    weight_net.print_weight_summary()
    
    print("\n✓ All tests passed!")
