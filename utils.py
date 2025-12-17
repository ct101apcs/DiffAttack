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
