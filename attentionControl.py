from typing import Union, Tuple
import torch
import abc


class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self.forward(attn[h // 2:], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self, res):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.res = res

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.res // 16) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + self.attention_store[key][i]
        self.step_store = self.get_empty_store()
        
        # Increment timestep counter for AttentionControlEdit
        if hasattr(self, 'current_timestep_idx'):
            self.current_timestep_idx += 1

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
        # Reset timestep tracking for AttentionControlEdit
        if hasattr(self, 'current_timestep_idx'):
            self.current_timestep_idx = 0
        
        # Reset timestep tracking for AttentionControlEdit
        if hasattr(self, 'current_timestep_idx'):
            self.current_timestep_idx = 0


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]], res):
        super(AttentionControlEdit, self).__init__(res)
        self.batch_size = 2
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.loss = 0
        self.criterion = torch.nn.MSELoss()
        self.timestep_loss_log = []
        self.current_timestep_idx = 0  # Track which timestep index we're at
        self.current_iter = 0  # Track current optimization iteration
    
    def set_timestep_index(self, timestep_idx):
        """Set the current timestep index for tracking."""
        self.current_timestep_idx = timestep_idx
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if not is_cross:
                """
                        ==========================================
                        ========= Self Attention Control =========
                        === Details please refer to Section 3.4 ==
                        ==========================================
                """
                timestep_loss = self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
                self.loss += timestep_loss
                
                # Log per-timestep loss (only once per timestep, at first attention layer)
                if self.cur_att_layer == 0:
                    if not hasattr(self, 'timestep_loss_log'):
                        self.timestep_loss_log = []
                    
                    self.timestep_loss_log.append({
                        'timestep': self.current_timestep_idx,
                        'loss': timestep_loss.item(),
                        'iteration': getattr(self, 'current_iter', 0)
                    })
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_self_attention(self, attn_base, att_replace):
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)


class AttentionControlEditLearnable(AttentionControlEdit):
    """
    Extension of AttentionControlEdit that supports learnable timestep weights.
    
    Instead of uniform weighting, applies learned weights to each timestep's loss
    contribution, allowing the model to learn which timesteps are most important
    for the adversarial attack.
    
    Based on empirical analysis showing:
    - Late timesteps (t=4) contribute 47-57% of loss (2.6x above uniform)
    - Early timesteps (t=0) contribute only 2-6% (0.3x of uniform)
    
    IMPORTANT: Use detach_weights=True to prevent gradient pathology where
    the optimizer learns to minimize loss by weighting DOWN high-loss timesteps.
    """
    
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]],
                 res,
                 weight_network=None,
                 use_learned_weights: bool = True,
                 detach_weights: bool = True):
        """
        Initialize the learnable attention controller.
        
        Args:
            num_steps: Total number of diffusion steps
            self_replace_steps: When to apply self-attention replacement
            res: Resolution of the input
            weight_network: TimestepWeightNetwork instance (optional)
            use_learned_weights: Whether to use learned weights (if False, uses uniform)
            detach_weights: If True, detach weights to prevent gradient pathology (RECOMMENDED)
        """
        super(AttentionControlEditLearnable, self).__init__(num_steps, self_replace_steps, res)
        
        self.weight_network = weight_network
        self.use_learned_weights = use_learned_weights
        self.detach_weights = detach_weights  # CRITICAL: prevents weight collapse
        
        # Cached weights for current iteration (set before denoising loop)
        self._cached_self_weights = None
        self._cached_cross_weights = None
        
        # Track weighted loss components for analysis
        self.weighted_self_loss = 0
        self.weighted_cross_loss = 0
        self.timestep_weighted_losses = []  # Track weighted losses per timestep
    
    def set_weights(self, w_self: torch.Tensor, w_cross: torch.Tensor):
        """
        Cache the current weights for use during the denoising loop.
        
        Should be called before each iteration's denoising loop with freshly
        computed weights from the weight network.
        
        Args:
            w_self: Self-attention weights tensor of shape [num_timesteps]
            w_cross: Cross-attention weights tensor of shape [num_timesteps]
        """
        self._cached_self_weights = w_self
        self._cached_cross_weights = w_cross
    
    def get_current_self_weight(self) -> torch.Tensor:
        """
        Get the weight for the current timestep (self-attention).
        
        Returns:
            Scalar weight for current timestep, or 1.0 if no weights set
        """
        if not self.use_learned_weights or self._cached_self_weights is None:
            return torch.tensor(1.0)
        
        timestep_idx = self.current_timestep_idx
        if timestep_idx < len(self._cached_self_weights):
            return self._cached_self_weights[timestep_idx]
        return torch.tensor(1.0)
    
    def get_current_cross_weight(self) -> torch.Tensor:
        """
        Get the weight for the current timestep (cross-attention).
        
        Returns:
            Scalar weight for current timestep, or 1.0 if no weights set
        """
        if not self.use_learned_weights or self._cached_cross_weights is None:
            return torch.tensor(1.0)
        
        timestep_idx = self.current_timestep_idx
        if timestep_idx < len(self._cached_cross_weights):
            return self._cached_cross_weights[timestep_idx]
        return torch.tensor(1.0)
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """
        Forward pass with weighted loss accumulation.
        
        Overrides parent to apply timestep-specific weights to the loss.
        """
        # Call grandparent's forward (AttentionStore) to store attention maps
        AttentionStore.forward(self, attn, is_cross, place_in_unet)
        
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            
            if not is_cross:
                """
                ==========================================
                ========= Self Attention Control =========
                === Details please refer to Section 3.4 ==
                ==========================================
                """
                timestep_loss = self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
                
                # Apply learned weight if enabled
                if self.use_learned_weights and self._cached_self_weights is not None:
                    weight = self.get_current_self_weight()
                    # Scale by num_timesteps to keep magnitude comparable to unweighted version
                    # (since weights sum to 1, total would be 1/N times original without scaling)
                    num_timesteps = len(self._cached_self_weights)
                    
                    # CRITICAL FIX: Detach weight to prevent gradient pathology!
                    # Without this, optimizer learns to minimize loss by weighting DOWN 
                    # high-loss timesteps, causing complete weight collapse to timestep 0.
                    if self.detach_weights:
                        weighted_loss = weight.detach() * num_timesteps * timestep_loss
                    else:
                        weighted_loss = weight * num_timesteps * timestep_loss
                    
                    self.loss += weighted_loss
                    
                    # Track weighted loss for analysis
                    if self.cur_att_layer == 0:
                        self.timestep_weighted_losses.append({
                            'timestep': self.current_timestep_idx,
                            'raw_loss': timestep_loss.item(),
                            'weight': weight.item() if hasattr(weight, 'item') else weight,
                            'weighted_loss': weighted_loss.item(),
                            'iteration': getattr(self, 'current_iter', 0)
                        })
                else:
                    # Original behavior: uniform weighting
                    self.loss += timestep_loss
                
                # Log per-timestep loss (only once per timestep, at first attention layer)
                if self.cur_att_layer == 0:
                    if not hasattr(self, 'timestep_loss_log'):
                        self.timestep_loss_log = []
                    
                    self.timestep_loss_log.append({
                        'timestep': self.current_timestep_idx,
                        'loss': timestep_loss.item(),
                        'iteration': getattr(self, 'current_iter', 0)
                    })
            
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def reset(self):
        """Reset the controller state for a new iteration."""
        super(AttentionControlEditLearnable, self).reset()
        self.weighted_self_loss = 0
        self.weighted_cross_loss = 0
        self.timestep_weighted_losses = []
    
    def get_weight_statistics(self) -> dict:
        """
        Get statistics about the weights used in the current iteration.
        
        Returns:
            Dict with weight statistics
        """
        if not self.timestep_weighted_losses:
            return {}
        
        import numpy as np
        
        raw_losses = [x['raw_loss'] for x in self.timestep_weighted_losses]
        weights = [x['weight'] for x in self.timestep_weighted_losses]
        weighted_losses = [x['weighted_loss'] for x in self.timestep_weighted_losses]
        timesteps = [x['timestep'] for x in self.timestep_weighted_losses]
        
        return {
            'timesteps': timesteps,
            'raw_losses': raw_losses,
            'weights': weights,
            'weighted_losses': weighted_losses,
            'total_raw_loss': sum(raw_losses),
            'total_weighted_loss': sum(weighted_losses)
        }


class AttentionControlEditFixedWeights(AttentionControlEdit):
    """
    Use fixed learned weights (no optimization during attack).
    Based on empirical learning from pilot study on small dataset.
    
    This approach is RECOMMENDED over learnable weights because:
    1. Learnable weights suffer from gradient pathology (weight collapse)
    2. Fixed schedule is stable and reproducible
    3. The learned schedule [0.064, 0.103, 0.158, 0.269, 0.406] was derived 
       from actual optimization on a small dataset
    
    Default schedule emphasizes late timesteps which contribute more to 
    self-attention loss based on empirical analysis.
    """
    
    # Learned schedule from pilot study (30 iterations on 10 images)
    DEFAULT_LEARNED_SCHEDULE = [0.064, 0.103, 0.158, 0.269, 0.406]
    
    # Alternative: initial increasing schedule based on loss analysis
    INITIAL_INCREASING_SCHEDULE = [0.05, 0.08, 0.13, 0.24, 0.50]
    
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]],
                 res,
                 weight_schedule: list = None,
                 schedule_type: str = 'learned'):
        """
        Initialize the fixed-weight attention controller.
        
        Args:
            num_steps: Total number of diffusion steps
            self_replace_steps: When to apply self-attention replacement
            res: Resolution of the input
            weight_schedule: Custom weight schedule (list of floats summing to ~1.0)
                            If None, uses schedule_type to select default
            schedule_type: 'learned' (default), 'increasing', or 'uniform'
        """
        super(AttentionControlEditFixedWeights, self).__init__(num_steps, self_replace_steps, res)
        
        if weight_schedule is not None:
            self.weight_schedule = weight_schedule
        elif schedule_type == 'learned':
            self.weight_schedule = self.DEFAULT_LEARNED_SCHEDULE.copy()
        elif schedule_type == 'increasing':
            self.weight_schedule = self.INITIAL_INCREASING_SCHEDULE.copy()
        elif schedule_type == 'uniform':
            # Will be set dynamically based on actual timesteps
            self.weight_schedule = None
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        
        self.schedule_type = schedule_type
        self.timestep_weighted_losses = []  # For tracking/analysis
    
    def get_current_weight(self) -> float:
        """Get the fixed weight for the current timestep."""
        if self.weight_schedule is None:
            return 1.0  # Uniform weighting
        
        timestep_idx = self.current_timestep_idx
        if timestep_idx < len(self.weight_schedule):
            return self.weight_schedule[timestep_idx]
        return 1.0
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """Forward pass with fixed weight application."""
        # Call grandparent's forward (AttentionStore) to store attention maps
        AttentionStore.forward(self, attn, is_cross, place_in_unet)
        
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            
            if not is_cross:
                """
                ==========================================
                ========= Self Attention Control =========
                === with Fixed Learned Weights ===========
                ==========================================
                """
                timestep_loss = self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
                
                # Apply fixed weight
                weight = self.get_current_weight()
                num_timesteps = len(self.weight_schedule) if self.weight_schedule else 1
                
                # Scale by num_timesteps to keep magnitude comparable
                weighted_loss = weight * num_timesteps * timestep_loss
                self.loss += weighted_loss
                
                # Track for analysis
                if self.cur_att_layer == 0:
                    self.timestep_weighted_losses.append({
                        'timestep': self.current_timestep_idx,
                        'raw_loss': timestep_loss.item(),
                        'weight': weight,
                        'weighted_loss': weighted_loss.item(),
                        'iteration': getattr(self, 'current_iter', 0)
                    })
                
                # Also log to timestep_loss_log for compatibility
                if self.cur_att_layer == 0:
                    if not hasattr(self, 'timestep_loss_log'):
                        self.timestep_loss_log = []
                    self.timestep_loss_log.append({
                        'timestep': self.current_timestep_idx,
                        'loss': timestep_loss.item(),
                        'iteration': getattr(self, 'current_iter', 0)
                    })
            
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def reset(self):
        """Reset the controller state for a new iteration."""
        super(AttentionControlEditFixedWeights, self).reset()
        self.timestep_weighted_losses = []
    
    def get_schedule_info(self) -> dict:
        """Get information about the weight schedule being used."""
        return {
            'schedule_type': self.schedule_type,
            'weights': self.weight_schedule,
            'num_timesteps': len(self.weight_schedule) if self.weight_schedule else 0
        }
