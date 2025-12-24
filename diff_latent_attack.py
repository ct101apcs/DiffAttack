import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from typing import Sequence, Tuple
from distances import LpDistance
import other_attacks
from typing import Optional

# --- SCIENTIFIC HELPER FUNCTIONS ---

def masked_max_entropy_loss(attn_map, mask=None, eps=1e-10):
    """
    Computes Shannon Entropy H(x) with LOCAL normalization.
    
    Fix: Normalizes probabilities WITHIN the mask.
    This forces the object attention to be uniform (flat) without forcing
    probability mass to leak into the background.
    """
    # attn_map: [Batch, Pixels, Tokens]
    # mask: [Batch, Pixels]
    
    if mask is not None:
        # Flatten mask to [Batch, Pixels]
        mask_flat = mask.view(mask.shape[0], -1)
        
        # 1. Apply mask FIRST (Zero out background attention)
        # We expand mask to [Batch, Pixels, 1] for broadcasting over tokens
        masked_attn = attn_map * mask_flat.unsqueeze(-1)
        
        # 2. Normalize probabilities ONLY within the mask
        # Sum over Pixels (dim=1) to get total attention mass per token
        total_mass = masked_attn.sum(dim=1, keepdim=True) + eps
        prob = masked_attn / total_mass
        
        # 3. Calculate Entropy: -p * log(p)
        # We only care about pixels inside the mask
        # Add eps inside log, but multiply by prob (which is 0 for bg) to keep bg entropy 0
        pixel_entropy = -prob * torch.log(prob + eps)
        
        # 4. Sum over tokens and pixels
        # Since background prob is 0, it contributes 0 entropy.
        total_entropy = pixel_entropy.sum()
        
        # Normalize by number of object pixels to keep scale consistent
        num_obj_pixels = mask_flat.sum().clamp(min=1.0)
        
        # Return negative entropy (maximize entropy)
        return -total_entropy / num_obj_pixels
    
    else:
        # Fallback to global entropy (Leaky)
        prob = attn_map / (attn_map.sum(dim=1, keepdim=True) + eps)
        pixel_entropy = -prob * torch.log(prob + eps)
        return -pixel_entropy.sum(dim=-1).mean()

def get_attention_with_grad(prompt: Sequence[str],
                            controller,
                            res: int,
                            from_where: Tuple[str, ...] = ("up", "down"),
                            is_cross: bool = True,
                            select: int = 0) -> torch.Tensor:
    """
    Retrieves attention maps with gradients attached for optimization.
    Interpolates all spatial resolutions to target res.
    """
    # Check attention_store first (populated after diffusion loop via between_steps)
    if hasattr(controller, 'attention_store') and len(controller.attention_store) > 0:
        attn_store = controller.attention_store
    elif hasattr(controller, 'step_store'):
        attn_store = controller.step_store
    else:
        attn_store = {}
    
    keys = [f"{w}_{'cross' if is_cross else 'self'}" for w in from_where]
    maps = []

    for key in keys:
        if key not in attn_store:
            continue
        for attn in attn_store[key]:
            if attn.dim() != 3:
                continue
            
            heads, q, k = attn.shape
            spatial = int(q ** 0.5)
            
            if spatial * spatial != q or spatial < 4:
                continue
            
            # Average over heads
            attn_avg = attn.mean(0)  # [q, k]
            
            # Reshape to spatial
            attn_spatial = attn_avg.reshape(spatial, spatial, k)  # [spatial, spatial, k]
            
            # Interpolate to target res
            attn_resized = torch.nn.functional.interpolate(
                attn_spatial.permute(2, 0, 1).unsqueeze(0),  # [1, k, spatial, spatial]
                size=(res, res),
                mode='bilinear',
                align_corners=False
            )  # [1, k, res, res]
            
            attn_resized = attn_resized.squeeze(0).permute(1, 2, 0)  # [res, res, k]
            attn_resized = attn_resized.reshape(res * res, k)  # [res*res, k]
            
            maps.append(attn_resized)

    if len(maps) == 0:
        return torch.zeros(res * res, 77, device='cuda', requires_grad=True)

    # Average across layers
    att = torch.stack(maps, dim=0).mean(0)
    return att

# ---------------------------------------------------------------------------

def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0

def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)

@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5, res=512):
    batch_size = 1
    max_length = 77
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(prompt[0], padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    model.scheduler.set_timesteps(num_inference_steps)
    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)
    all_latents = [latents]
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))
        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred
        all_latents.append(latents)
    return latents, all_latents

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            if hasattr(self, 'reshape_heads_to_batch_dim'):
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)
            else:
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            if hasattr(self, 'reshape_batch_dim_to_heads'):
                out = self.reshape_batch_dim_to_heads(out)
            else:
                out = self.batch_to_head_dim(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ in ['CrossAttention', 'Attention']:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]: cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]: cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]: cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

def reset_attention_control(model):
    def ca_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            if hasattr(self, 'reshape_heads_to_batch_dim'):
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)
            else:
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            if hasattr(self, 'reshape_batch_dim_to_heads'):
                out = self.reshape_batch_dim_to_heads(out)
            else:
                out = self.batch_to_head_dim(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out
        return forward

    def register_recr(net_):
        if net_.__class__.__name__ in ['CrossAttention', 'Attention']:
            net_.forward = ca_forward(net_)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]: register_recr(net[1])
        elif "up" in net[0]: register_recr(net[1])
        elif "mid" in net[0]: register_recr(net[1])

def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

@torch.enable_grad()
def diffattack(
        model,
        label,
        controller,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        image=None,
        model_name="inception",
        save_path=r"C:\Users\PC\Desktop\output",
        res=224,
        start_step=15,
        iterations=30,
        verbose=True,
        topN=1,
        args=None
):
    if args.dataset_name == "imagenet_compatible":
        from dataset_caption import imagenet_label
    elif args.dataset_name == "cub_200_2011":
        from dataset_caption import CUB_label as imagenet_label
    elif args.dataset_name == "standford_car":
        from dataset_caption import stanfordCar_label as imagenet_label
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()
    reset_attention_control(model)
    controller.reset()
    
    label = torch.from_numpy(label).long().cuda()
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)
    classifier = other_attacks.model_selection(model_name).eval().cuda()
    classifier.requires_grad_(False)

    height = width = res
    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = (torch.from_numpy(test_image).unsqueeze(0).float().contiguous())

    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))
    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())
    _, pred_labels = pred.topk(topN, largest=True, sorted=True)
    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, topN)])
    prompt = [imagenet_label.refined_Label[label.item()] + " " + target_prompt] * 2
    true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])

    latent, inversion_latents = ddim_reverse_sample(image, prompt, model, num_inference_steps, 0, res=height)
    inversion_latents = inversion_latents[::-1]
    init_prompt = [prompt[0]]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    # Optimize unconditional embeddings
    max_length = 77
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(init_prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)
    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
    loss_func = torch.nn.MSELoss()
    context = torch.cat([uncond_embeddings, text_embeddings])

    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
            loss.backward()
            optimizer.step()
            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)
        with torch.no_grad():
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    uncond_embeddings.requires_grad_(False)
    register_attention_control(model, controller)
    batch_size = len(prompt)
    text_input = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]

    original_latent = latent.clone()
    latent.requires_grad_(True)
    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    # === PATCHED FORWARD WITH STRUCTURE LOSS ===
    controller._detach_attention = True
    
    def patched_forward(attn, is_cross, place_in_unet):
        """
        NOTE: This function receives attention that has ALREADY been sliced by
        AttentionControl.__call__ (line 26 in attentionControl.py): attn[h // 2:]
        So 'attn' here contains only the second half of attention heads.
        For batch_size=2 (clean, adv), we receive heads from BOTH images combined.
        """
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        q = attn.shape[1]
        spatial = int(q ** 0.5)
        
        # 1. Structure Loss Calculation (Restored!)
        # Check conditions from original AttentionControlEdit
        # NOTE: attn shape here is [batch*heads_per_image, q, k] where batch=2
        # The attention has been sliced, so we need to split by batch, not by halving heads
        if not is_cross and (controller.num_self_replace[0] <= controller.cur_step < controller.num_self_replace[1]):
            # For batch_size=2, attn contains heads from both clean and adv images
            # Number of heads per image = total_heads / batch_size
            batch_size = 2  # clean and adversarial
            total_heads = attn.shape[0]
            
            # Only process if we have enough heads to split
            if total_heads >= batch_size and total_heads % batch_size == 0:
                heads_per_image = total_heads // batch_size
                # Reshape to [batch, heads_per_image, q, k]
                attn_reshaped = attn.reshape(batch_size, heads_per_image, *attn.shape[1:])
                attn_base = attn_reshaped[0]    # Clean image's attention
                attn_replace = attn_reshaped[1] # Adversarial image's attention
                
                # We need gradients for this!
                if not getattr(controller, '_detach_attention', True):
                    # Check if gradients are required
                    if attn_replace.requires_grad:
                        loss = torch.nn.MSELoss()(attn_replace, attn_base.detach())
                        controller.loss += loss

        # 2. Attention Storage
        if spatial * spatial == q and 4 <= spatial <= 32:
            if getattr(controller, '_detach_attention', True):
                controller.step_store[key].append(attn.detach())
            else:
                controller.step_store[key].append(attn)
        return attn
    
    controller.forward = patched_forward
    print("✓ Patched controller.forward (Structure Loss Restored)")

    # --- Pre-calculate Mask ---
    apply_mask = True
    attention_mask_flat = None
    MASK_SPATIAL_RES = 16

    if apply_mask:
        with torch.no_grad():
            controller.reset()
            controller.step_store = controller.get_empty_store()
            controller.attention_store = {}
            
            latents_mask = torch.cat([original_latent, original_latent])
            for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
                latents_mask = diffusion_step(model, latents_mask, context[ind], t, guidance_scale)
            
            attn_store = controller.attention_store if len(controller.attention_store) > 0 else controller.step_store
            collected_maps = []
            
            for location in ["up", "down", "mid"]:
                key = f"{location}_cross"
                if key not in attn_store: continue
                for attn in attn_store[key]:
                    if not isinstance(attn, torch.Tensor): continue
                    if attn.dim() == 3:
                        heads, q, k = attn.shape
                        spatial = int(q ** 0.5)
                        if spatial * spatial != q or spatial < 4: continue
                        
                        attn_avg = attn.mean(0)
                        obj_start, obj_end = 1, len(true_label) - 1
                        if obj_end <= obj_start: obj_end = min(k, 5)
                        
                        obj_attn = attn_avg[:, obj_start:obj_end].mean(-1)
                        obj_attn_2d = obj_attn.reshape(spatial, spatial)
                        
                        obj_attn_resized = torch.nn.functional.interpolate(
                            obj_attn_2d.unsqueeze(0).unsqueeze(0).float(),
                            size=(MASK_SPATIAL_RES, MASK_SPATIAL_RES),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                        collected_maps.append(obj_attn_resized.to(model.device))
            
            if collected_maps:
                mask = torch.stack(collected_maps).mean(0)
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                threshold = torch.quantile(mask.flatten(), 0.70)
                attention_mask = (mask > threshold).float()
                print(f"Mask generated. Coverage: {attention_mask.mean():.1%}")
                attention_mask_flat = attention_mask.flatten().unsqueeze(0)
            else:
                print("ERROR: No attention maps collected!")
                apply_mask = False
            controller.reset()

    # --- Main Loop ---
    controller._detach_attention = False # Enable grads for structure & entropy
    
    pbar = tqdm(range(iterations), desc="Iterations")
    
    for iter_idx, _ in enumerate(pbar):
        controller.loss = 0
        controller.reset()
        latents = torch.cat([original_latent, latent])
        
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        # 1. Get Attention Map with Gradients
        after_attention_map = get_attention_with_grad(
            prompt, controller, MASK_SPATIAL_RES, ("up", "down"), True, select=1
        )
        after_attention_map = after_attention_map.unsqueeze(0) 

        # 2. Slice Object Tokens
        after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]

        # 3. MASKED ENTROPY LOSS (LOCALLY NORMALIZED)
        # Weight can be adjusted. If entropy is smaller due to local norm, maybe increase weight.
        # Entropy of uniform distribution on object (e.g. 100 pixels) is ln(100) ~ 4.6.
        # Normalized by num_pixels ~ 4.6.
        neg_entropy = masked_max_entropy_loss(after_true_label_attention_map, mask=attention_mask_flat)
        entropy_loss = neg_entropy * args.cross_attn_loss_weight

        # 4. Standard Classifier & Structure Losses
        if apply_mask:
             init_mask = torch.nn.functional.interpolate(
                attention_mask.unsqueeze(0).unsqueeze(0).to(init_image.device),
                init_image.shape[-2:], mode="nearest"
             )
        else:
             init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).to(init_image.device)

        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (1 - init_mask) * init_image
        
        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        if args.dataset_name != "imagenet_compatible":
            pred = classifier(out_image) / 10
        else:
            pred = classifier(out_image)

        attack_loss = - cross_entro(pred, label) * args.attack_loss_weight
        self_attn_loss = controller.loss * args.self_attn_loss_weight # Now this should be non-zero
        
        loss = self_attn_loss + attack_loss + entropy_loss

        if verbose:
            pbar.set_postfix_str(
                f"atk: {attack_loss.item():.3f} "
                f"ent: {entropy_loss.item():.3f} "
                f"str: {self_attn_loss.item():.3f} "
                f"loss: {loss.item():.3f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        controller.loss = 0
        controller.reset()
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
            1 - init_mask) * init_image
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    out_image = out_image.float().contiguous()
    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    logit = torch.nn.Softmax()(pred)
    print("after_pred:", pred_label, logit[0, pred_label[0]])
    print("after_true:", label, logit[0, label[0]])

    image = latent2image(model.vae, latents.detach())
    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + (
            1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
    image = (perturbed * 255).astype(np.uint8)
    view_images(np.concatenate([real, perturbed]) * 255, show=False,
                save_path=save_path + "_diff_{}_image_{}.png".format(model_name,
                                                                     "ATKSuccess" if pred_accuracy == 0 else "Fail"))
    view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    print("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_relative.png")

    reset_attention_control(model)
    return image[0], 0, 0