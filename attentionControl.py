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
    def __init__(self, res, store_res_threshold=32):
        """
        Args:
            res: Image resolution (e.g., 224 for classifier, 512 for diffusion)
            store_res_threshold: Minimum spatial resolution to store (default 32 -> stores 32x32 and smaller)
        """
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.res = res
        # Store attention maps up to 32x32 (1024 tokens) to get good spatial detail
        # while avoiding memory issues from 64x64 (4096 tokens)
        self.max_tokens_to_store = store_res_threshold ** 2

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        q = attn.shape[1]
        
        # Store attention maps with spatial resolution up to threshold
        # This captures 8x8, 16x16, 32x32 but skips 64x64
        if q <= self.max_tokens_to_store:
            self.step_store[key].append(attn.detach())  # .detach() to save memory
        
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = {k: list(v) for k, v in self.step_store.items()}
        else:
            for key in self.attention_store:
                # Defensive check: if sizes don't match, reset and start fresh
                # This can happen when processing a new image with different batch size
                if len(self.step_store[key]) != len(self.attention_store[key]):
                    self.attention_store = {k: list(v) for k, v in self.step_store.items()}
                    self.step_store = self.get_empty_store()
                    return
                for i in range(len(self.attention_store[key])):
                    # Check tensor shapes match before accumulating
                    if self.step_store[key][i].shape != self.attention_store[key][i].shape:
                        # Shape mismatch - reset to fresh state with current step data
                        self.attention_store = {k: list(v) for k, v in self.step_store.items()}
                        self.step_store = self.get_empty_store()
                        return
                    self.attention_store[key][i] = self.step_store[key][i] + self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        if self.cur_step == 0:
            return self.step_store  # Return current step if no averaging yet
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] 
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


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
                self.loss += self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_self_attention(self, attn_base, att_replace):
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
