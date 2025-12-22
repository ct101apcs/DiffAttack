from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import other_attacks
from typing import Sequence, Tuple


def max_entropy_loss(attn_map, eps=1e-10, debug=False, iteration=None):
    """
    Calculates the Negative Entropy of the spatial attention map.
    Minimizing this value MAXIMIZES the Entropy (uncertainty).
    
    Args:
        attn_map: Tensor of shape [Batch, Spatial, Tokens]
    """
    # 1. Flatten spatial dimensions (Batch, Spatial_Pixels, Tokens)
    # We want the 'Cat' token to look at the whole image uniformly.
    b, s, t = attn_map.shape
    
    # DEBUG: Check if attention map is empty or all zeros
    if debug and iteration is not None and iteration % 5 == 0:
        print(f"\n[DEBUG Iter {iteration}] Attention Map Analysis:")
        print(f"  Shape: {attn_map.shape} (batch={b}, spatial={s}, tokens={t})")
        print(f"  Min/Max/Mean: {attn_map.min().item():.6f}/{attn_map.max().item():.6f}/{attn_map.mean().item():.6f}")
        print(f"  Sum: {attn_map.sum().item():.6f}")
        print(f"  All zeros: {(attn_map == 0).all().item()}")
    
    # 2. Normalize spatial dimension to create a valid Probability Distribution P(x)
    # P(x) = Attention at pixel x / Total Attention
    # We compute this PER TOKEN.
    spatial_sum = attn_map.sum(dim=1, keepdim=True)
    prob = attn_map / (spatial_sum + eps)
    
    if debug and iteration is not None and iteration % 5 == 0:
        print(f"  Spatial sum range: {spatial_sum.min().item():.6f} to {spatial_sum.max().item():.6f}")
        print(f"  Prob range: {prob.min().item():.6f} to {prob.max().item():.6f}")
        print(f"  Max prob per token (concentration): {prob.max(dim=1)[0].mean().item():.6f}")
    
    # 3. Calculate Entropy H(P) = - sum(p * log(p))
    # We want to MAXIMIZE H, so we MINIMIZE: sum(p * log(p))
    entropy = (prob * torch.log(prob + eps)).sum(dim=1)
    
    if debug and iteration is not None and iteration % 5 == 0:
        print(f"  Entropy per token range: {entropy.min().item():.6f} to {entropy.max().item():.6f}")
        print(f"  Final entropy: {entropy.mean().item():.6f}")
    
    # Average over batch and tokens
    return entropy.mean()


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

def get_attention_with_grad(prompt: Sequence[str],
                            controller,
                            res: int,
                            from_where: Tuple[str, ...] = ("up", "down"),
                            is_cross: bool = True,
                            select: int = 0,
                            debug: bool = False) -> torch.Tensor:
    """
    Gradient-preserving version of aggregate_attention.
    Does NOT use @torch.no_grad().
    """
    # IMPORTANT: Use get_average_attention() if available (averaged over diffusion steps)
    # Otherwise fall back to attention_store
    attn_store = controller.get_average_attention() if hasattr(controller, 'get_average_attention') else getattr(controller, 'attention_store', {})
    keys = [f"{w}_{'cross' if is_cross else 'self'}" for w in from_where]
    maps = []
    bsz = 2  # Standard unet batch size (uncond + cond)

    if debug:
        print(f"\n[DEBUG] get_attention_with_grad:")
        print(f"  Using: {'get_average_attention()' if hasattr(controller, 'get_average_attention') else 'attention_store'}")
        print(f"  Looking for keys: {keys}")
        print(f"  Available keys: {list(attn_store.keys())}")
        print(f"  Target resolution: {res}x{res} = {res*res}")
        
        # Show what resolutions are actually available
        available_resolutions = set()
        for key in attn_store.keys():
            if isinstance(attn_store.get(key), list):
                for attn in attn_store.get(key, []):
                    if hasattr(attn, 'shape') and len(attn.shape) > 1:
                        q = attn.shape[1]
                        available_resolutions.add(q)
        if available_resolutions:
            print(f"  Available q values: {sorted(available_resolutions)}")
            print(f"  Available sqrt(q): {sorted([int(q**0.5) for q in available_resolutions if q > 0])}")

    for key in keys:
        if key not in attn_store:
            if debug:
                print(f"  Key '{key}' not found in attention store")
            continue
        
        attn_list = attn_store.get(key, [])
        if debug:
            print(f"  Processing key '{key}': {len(attn_list)} attention maps")
        
        for idx, attn in enumerate(attn_list):
            if not isinstance(attn, torch.Tensor):
                continue
                
            # attn shape: [batch*heads, query, key]
            bh, q, k = attn.shape
            h = bh // bsz
            
            if debug and idx < 3:
                print(f"    Map {idx}: shape={attn.shape}, q={q}, sqrt(q)={int(q**0.5) if q > 0 else 0}, target={res*res}")
            
            # Reshape and average heads
            attn = attn.reshape(bsz, h, q, k)
            attn = attn.mean(1) # [bsz, q, k]

            # Filter by resolution - exact match first
            if q == res * res:
                # Select the adversarial prompt index (usually 1)
                att_sel = attn[select] # [q, k]
                # Reshape to spatial
                att_sel = att_sel.reshape(res, res, k)
                maps.append(att_sel)
                if debug:
                    print(f"    ✓ Added map {idx}: final shape {att_sel.shape} (exact match)")
            else:
                # Try to reshape and interpolate from different resolution
                sq = int(q ** 0.5)
                if sq * sq == q:  # Perfect square, can reshape spatially
                    att_sel = attn[select].reshape(sq, sq, k)
                    # Resize to target resolution using bilinear interpolation
                    att_chw = att_sel.permute(2, 0, 1).unsqueeze(0)  # [1, k, sq, sq]
                    att_resized = torch.nn.functional.interpolate(
                        att_chw, size=(res, res), mode='bilinear', align_corners=False
                    )
                    att_resized = att_resized.squeeze(0).permute(1, 2, 0)  # [res, res, k]
                    maps.append(att_resized)
                    if debug:
                        print(f"    ✓ Added map {idx}: resized from {sq}x{sq} to {res}x{res}")

    if debug:
        print(f"  Total maps collected: {len(maps)}")

    if len(maps) == 0:
        # Infer device from attention store if available, otherwise use cuda
        device = 'cuda'
        for key in attn_store:
            if isinstance(attn_store.get(key), list) and len(attn_store.get(key, [])) > 0:
                device = attn_store[key][0].device
                break
        if debug:
            print(f"  ⚠️ WARNING: No attention maps found! Returning zeros.")
        return torch.zeros(res, res, 77, device=device, requires_grad=True)

    # Stack and average across layers
    att = torch.stack(maps, dim=0).mean(0) # [res, res, Tokens]
    if debug:
        print(f"  Final attention map shape: {att.shape}")
        print(f"  Range: [{att.min().item():.6f}, {att.max().item():.6f}]")
    return att

@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


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

    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())

    _, pred_labels = pred.topk(topN, largest=True, sorted=True)

    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, topN)])
    prompt = [imagenet_label.refined_Label[label.item()] + " " + target_prompt] * 2
    print("prompt generate: ", prompt[0], "\tlabels: ", pred_labels.cpu().numpy().tolist())

    true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])
    target_label = model.tokenizer.encode(target_prompt)
    print("decoder: ", true_label, target_label)

    """
            ==========================================
            ============ DDIM Inversion ==============
            === Details please refer to Appendix B ===
            ==========================================
    """
    latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                    num_inference_steps,
                                                    0, res=height)
    inversion_latents = inversion_latents[::-1]

    init_prompt = [prompt[0]]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    """
            ===============================================================================
            === Good initial reconstruction by optimizing the unconditional embeddings ====
            ======================= Details please refer to Section 3.4 ===================
            ===============================================================================
    """
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
    loss_func = torch.nn.MSELoss()

    context = torch.cat([uncond_embeddings, text_embeddings])

    #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
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

    """
            ==========================================
            ============ Latents Attack ==============
            ==== Details please refer to Section 3 ===
            ==========================================
    """

    uncond_embeddings.requires_grad_(False)

    register_attention_control(model, controller)

    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]

    original_latent = latent.clone()

    latent.requires_grad_(True)

    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    #  “Pseudo” Mask for better Imperceptibility, yet sacrifice the transferability. Details please refer to Appendix D.
    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    pbar = tqdm(range(iterations), desc="Iterations")
    
    for iter_idx, _ in enumerate(pbar):
        controller.loss = 0
        controller.reset()
        latents = torch.cat([original_latent, latent])
        
        # 1. Diffusion Steps
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        # ------------------------------------------------------
        # NEW CODE: USE GRADIENT-AWARE FUNCTION
        # ------------------------------------------------------
        
        # Enable debug on first iteration and every 10th iteration
        debug_mode = (iter_idx == 0 or iter_idx % 10 == 0)
        
        # Use the new function we defined at the top of this file
        after_attention_map = get_attention_with_grad(
            prompt, controller, args.res // 32, ("up", "down"), True, select=1, debug=debug_mode
        )

        # Slice to get object tokens
        after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]

        # Calculate Max Entropy Loss
        # We want to MAXIMIZE entropy, so we MINIMIZE the negative entropy returned by the function.
        # We normalize by numel to keep gradients stable.
        neg_entropy = max_entropy_loss(after_true_label_attention_map, debug=debug_mode, iteration=iter_idx)
        
        # Loss = (Negative Entropy) * Weight
        # If entropy increases, neg_entropy becomes more negative (e.g., -4.0).
        # We want to minimize this.
        entropy_loss = neg_entropy * args.cross_attn_loss_weight

        # ------------------------------------------------------

        # Generate output image for classifier
        # (You can use the 'aggregate_attention' from utils here for the MASK only, since mask doesn't need grad)
        with torch.no_grad():
             before_attn_vis = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False)
        
        before_true_label_map_vis = before_attn_vis[:, :, 1: len(true_label) - 1]
        
        if init_mask is None:
             init_mask = torch.nn.functional.interpolate(
                (before_true_label_map_vis.mean(-1) / before_true_label_map_vis.mean(-1).max()).unsqueeze(0).unsqueeze(0),
                init_image.shape[-2:], mode="bilinear").clamp(0, 1)
             if hard_mask:
                 init_mask = init_mask.gt(0.5).float()
        
        # Decode image
        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (1 - init_mask) * init_image
        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        # Classifier Loss
        if args.dataset_name != "imagenet_compatible":
            pred = classifier(out_image) / 10
        else:
            pred = classifier(out_image)
        attack_loss = - cross_entro(pred, label) * args.attack_loss_weight

        # Structure Loss
        self_attn_loss = controller.loss * args.self_attn_loss_weight

        # Total Loss
        loss = self_attn_loss + attack_loss + entropy_loss

        if verbose:
            pbar.set_postfix_str(
                f"atk: {attack_loss.item():0.8f} "
                f"ent: {entropy_loss.item():0.8f} "
                f"str: {self_attn_loss.item():0.8f} "
                f"loss: {loss.item():0.8f}")

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

    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    logit = torch.nn.Softmax()(pred)
    print("after_pred:", pred_label, logit[0, pred_label[0]])
    print("after_true:", label, logit[0, label[0]])

    """
            ==========================================
            ============= Visualization ==============
            ==========================================
    """

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

    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_absolute.png")

    reset_attention_control(model)

    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=args.res // 32, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionBefore.jpg".format(save_path))
    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=args.res // 32, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionAfter.jpg".format(save_path), select=1)
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionBefore.jpg".format(save_path))
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionAfter.jpg".format(save_path), select=1)

    return image[0], 0, 0

