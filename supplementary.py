import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
import glob
from PIL import Image
import random
import sys
from natsort import ns, natsorted
import argparse
from other_attacks import model_transfer

parser = argparse.ArgumentParser(description='Baseline Lp-norm Adversarial Attacks')

# Core arguments
parser.add_argument('--save_dir', default="supplementary", type=str)
parser.add_argument('--images_root', default="datasets/imagenet-compatible/images", type=str)
parser.add_argument('--label_path', default="datasets/imagenet-compatible/labels.txt", type=str)
parser.add_argument('--is_test', default=False, type=bool)

# Attack selection (accepts flexible aliases, canonicalized later)
parser.add_argument('--attack_method', default="MI-FGSM", type=str,
                    help='Attack method: FGSM | MI-FGSM | DI-FGSM | TI-FGSM | PI-FGSM | S2I-FGSM (aliases allowed)')

# Core L_inf parameters (Appendix F)
parser.add_argument('--iterations', default=10, type=int,
                    help='Steps (paper: 10 for all I-FGSM based methods)')

parser.add_argument('--epsilon', default=16/255, type=float,
                    help='Maximum perturbation (paper: 16/255)')

parser.add_argument('--step_size', default=1.6/255, type=float,
                    help='Step size (paper: 1.6/255)')

# Method-specific parameters (Appendix F)
parser.add_argument('--decay', default=1.0, type=float,
                    help='MI-FGSM decay factor (paper: 1.0)')

parser.add_argument('--prob', default=0.5, type=float,
                    help='DI-FGSM transformation probability (paper: 0.5)')

parser.add_argument('--kernel_size', default=7, type=int,
                    help='TI-FGSM kernel size (paper: 7)')

parser.add_argument('--amplification', default=10, type=int,
                    help='PI-FGSM amplification factor (paper: 10)')

parser.add_argument('--s2i_num_copies', default=20, type=int,
                    help='S2I-FGSM inner iterations (paper: 20)')

parser.add_argument('--s2i_variance', default=0.5, type=float,
                    help='S2I-FGSM tuning factor (paper: 0.5)')

parser.add_argument('--s2i_std', default=16, type=int,
                    help='S2I-FGSM standard deviation (paper: 16)')

# Model arguments (accepts flexible aliases, canonicalized later)
parser.add_argument('--model_name', default="resnet50", type=str,
                    help='Surrogate model: resnet|resnet50, vgg|vgg19, vit|vitb16, swin|swinb, inception_v3, mobilenet_v2')
parser.add_argument('--res', default=224, type=int)
parser.add_argument('--dataset_name', default="imagenet_compatible", type=str)

# (Note) parameters defined above already match paper settings; avoid duplicates


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(42)


def load_surrogate_model(model_name):
    """Load surrogate model"""
    print(f"Loading surrogate model: {model_name}")

    name = canonicalize_model_name(model_name)
    if name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif name == "vit_b_16":
        # torchvision vit
        try:
            model = models.vit_b_16(pretrained=True)
        except TypeError:
            model = models.vit_b_16(weights=getattr(models, 'ViT_B_16_Weights').IMAGENET1K_V1)
    elif name == "swin_b":
        # torchvision swin
        try:
            model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        except AttributeError:
            model = models.swin_b(pretrained=True)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False  # Disable auxiliary outputs
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
    model = model.cuda().eval()
    
    # Disable gradient computation for model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def input_diversity(x, prob=0.5):
    """
    Input Diversity for DI-FGSM
    Reference: Xie et al., 2019
    """
    if random.random() > prob:
        return x
    
    img_size = x.shape[-1]
    img_resize = int(img_size * random.uniform(0.875, 1.0))
    
    x_resize = F.interpolate(x, size=(img_resize, img_resize), 
                            mode='bilinear', align_corners=False)
    
    # Random padding
    pad_left = random.randint(0, img_size - img_resize)
    pad_top = random.randint(0, img_size - img_resize)
    pad_right = img_size - img_resize - pad_left
    pad_bottom = img_size - img_resize - pad_top
    
    x_padded = F.pad(x_resize, (pad_left, pad_right, pad_top, pad_bottom))
    
    return x_padded


def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    """Apply standard ImageNet normalization to a BCHW tensor in [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def canonicalize_attack_method(name: str) -> str:
    t = name.strip().lower().replace("_", "-")
    mapping = {
        "fgsm": "FGSM",
        "mi-fgsm": "MI-FGSM",
        "mifgsm": "MI-FGSM",
        "mi": "MI-FGSM",
        "di-fgsm": "DI-FGSM",
        "difgsm": "DI-FGSM",
        "di": "DI-FGSM",
        "ti-fgsm": "TI-FGSM",
        "tifgsm": "TI-FGSM",
        "ti": "TI-FGSM",
        "pi-fgsm": "PI-FGSM",
        "pifgsm": "PI-FGSM",
        "pi": "PI-FGSM",
        "s2i-fgsm": "S2I-FGSM",
        "s2ifgsm": "S2I-FGSM",
        "s2i": "S2I-FGSM",
    }
    if t in mapping:
        return mapping[t]
    raise ValueError(f"Unsupported attack method '{name}'. Try one of: FGSM, MI-FGSM, DI-FGSM, TI-FGSM, PI-FGSM, S2I-FGSM")


def canonicalize_model_name(name: str) -> str:
    n = name.strip().lower().replace(" ", "").replace("_", "-")
    aliases = {
        "resnet50": {"resnet", "resnet50", "rn50", "resnet-50", "restnet50"},
        "vgg19": {"vgg", "vgg19", "vgg-19"},
        "vit_b_16": {"vit", "vitb16", "vit-b16", "vit-b-16", "vit-b/16", "vit_b_16"},
        "swin_b": {"swin", "swinb", "swin-b", "swin_b"},
        "inception_v3": {"inception", "inceptionv3", "inception-v3", "inception_v3"},
        "mobilenet_v2": {"mobilenet", "mobilenetv2", "mobilenet-v2", "mobilenet_v2"},
    }
    for canon, opts in aliases.items():
        if n in opts:
            return canon
    # if the input was already a torchvision-acceptable name, pass it through
    return name


def gkern(kernlen=7, nsig=3):
    """Generate Gaussian kernel for TI-FGSM"""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = np.exp(-0.5 * x**2)
    kern2d = np.outer(kern1d, kern1d)
    kernel = kern2d / kern2d.sum()
    kernel = kernel.astype(np.float32)
    
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    return torch.from_numpy(gaussian_kernel).cuda()


def fgsm_attack(model, image, label, epsilon=16/255):
    """Basic FGSM Attack"""
    image.requires_grad = True
    
    output = model(_imagenet_norm(image))
    loss = nn.CrossEntropyLoss()(output, label)
    
    loss.backward()
    grad = image.grad.data
    
    # Generate adversarial example
    adv_image = image + epsilon * grad.sign()
    adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image.detach()


def mi_fgsm_attack(model, image, label, epsilon=16/255, step_size=1.6/255, 
                   iterations=10, decay=1.0):
    """
    MI-FGSM Attack
    Reference: Dong et al., 2018 - Boosting Adversarial Attacks with Momentum
    """
    delta = torch.zeros_like(image).cuda()
    momentum = torch.zeros_like(image).cuda()
    
    for i in range(iterations):
        delta.requires_grad = True
        
        adv_image = image + delta
        adv_image = torch.clamp(adv_image, 0, 1)
        
        output = model(_imagenet_norm(adv_image))
        loss = nn.CrossEntropyLoss()(output, label)
        
        loss.backward()
        grad = delta.grad.data
        
        # Normalize gradient
        grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
        
        # Update momentum
        momentum = decay * momentum + grad
        
        # Update delta
        delta.data = delta.data + step_size * momentum.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad = None
    
    adv_image = image + delta
    adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image.detach()


def di_fgsm_attack(model, image, label, epsilon=16/255, step_size=1.6/255,
                   iterations=10, decay=1.0, prob=0.5):
    """
    DI-FGSM Attack (MI-FGSM + Input Diversity)
    Reference: Xie et al., 2019 - Improving Transferability with Input Diversity
    """
    delta = torch.zeros_like(image).cuda()
    momentum = torch.zeros_like(image).cuda()
    
    for i in range(iterations):
        delta.requires_grad = True
        
        adv_image = image + delta
        adv_image = torch.clamp(adv_image, 0, 1)
        
        # Apply input diversity
        adv_image_diverse = input_diversity(adv_image, prob=prob)
        
        output = model(_imagenet_norm(adv_image_diverse))
        loss = nn.CrossEntropyLoss()(output, label)
        
        loss.backward()
        grad = delta.grad.data
        
        # Normalize gradient
        grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
        
        # Update momentum
        momentum = decay * momentum + grad
        
        # Update delta
        delta.data = delta.data + step_size * momentum.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad = None
    
    adv_image = image + delta
    adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image.detach()


def ti_fgsm_attack(model, image, label, epsilon=16/255, step_size=1.6/255,
                   iterations=10, decay=1.0, kernel_size=7):
    """
    TI-FGSM Attack (Translation-Invariant Attack)
    Reference: Dong et al., 2019 - Evading Defenses with Translation-Invariant Attacks
    """
    delta = torch.zeros_like(image).cuda()
    momentum = torch.zeros_like(image).cuda()
    
    # Generate Gaussian kernel
    gaussian_kernel = gkern(kernel_size, 3)
    
    for i in range(iterations):
        delta.requires_grad = True
        
        adv_image = image + delta
        adv_image = torch.clamp(adv_image, 0, 1)
        
        output = model(_imagenet_norm(adv_image))
        loss = nn.CrossEntropyLoss()(output, label)
        
        loss.backward()
        grad = delta.grad.data
        
        # Convolve gradient with Gaussian kernel
        grad = F.conv2d(grad, gaussian_kernel, padding=kernel_size//2, groups=3)
        
        # Normalize gradient
        grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
        
        # Update momentum
        momentum = decay * momentum + grad
        
        # Update delta
        delta.data = delta.data + step_size * momentum.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad = None
    
    adv_image = image + delta
    adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image.detach()


def pi_fgsm_attack(model, image, label, epsilon=16/255, step_size=1.6/255,
                   iterations=10, amplification=10):
    """
    PI-FGSM Attack (Patch-wise Iterative Attack)
    Reference: Gao et al., 2020 - Patch-wise Attack for Fooling Deep Neural Networks
    """
    delta = torch.zeros_like(image).cuda()
    amplification_factor = amplification
    
    for i in range(iterations):
        # Amplify perturbation
        delta_amplified = delta * amplification_factor
        delta_amplified.requires_grad = True
        
        adv_image = image + delta_amplified
        
        output = model(_imagenet_norm(adv_image))
        loss = nn.CrossEntropyLoss()(output, label)
        
        loss.backward()
        grad = delta_amplified.grad.data
        
        # Update delta
        delta = delta + step_size * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
    
    adv_image = image + delta
    adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image.detach()


def run_baseline_attack(image, label, model, args):
    """Run the selected baseline attack"""
    
    # Preprocess image
    if isinstance(image, Image.Image):
        image = image.resize((args.res, args.res), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
    
    image = torch.from_numpy(image).cuda()
    label = torch.from_numpy(label).cuda()
    
    # Get clean prediction (normalized)
    with torch.no_grad():
        clean_output = model(_imagenet_norm(image))
        clean_pred = clean_output.argmax(dim=1)
        clean_acc = (clean_pred == label).float().mean().item()
    
    # Run attack
    method = canonicalize_attack_method(args.attack_method)

    if method == "FGSM":
        adv_image = fgsm_attack(model, image, label, args.epsilon)
    elif method == "MI-FGSM":
        adv_image = mi_fgsm_attack(model, image, label, args.epsilon, 
                                   args.step_size, args.iterations, args.decay)
    elif method == "DI-FGSM":
        adv_image = di_fgsm_attack(model, image, label, args.epsilon,
                                   args.step_size, args.iterations, args.decay, args.prob)
    elif method == "TI-FGSM":
        adv_image = ti_fgsm_attack(model, image, label, args.epsilon,
                                   args.step_size, args.iterations, args.decay, args.kernel_size)
    elif method == "PI-FGSM":
        adv_image = pi_fgsm_attack(model, image, label, args.epsilon,
                                   args.step_size, args.iterations, args.amplification)
    else:
        raise ValueError(f"Attack method {args.attack_method} not supported")
    
    # Get adversarial prediction
    with torch.no_grad():
        adv_output = model(_imagenet_norm(adv_image))
        adv_pred = adv_output.argmax(dim=1)
        adv_acc = (adv_pred == label).float().mean().item()
    
    # Convert to image format
    adv_image_np = (adv_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)
    
    return adv_image_np, clean_acc, adv_acc

def print_paper_settings():
    """Print the exact settings from the paper for reference"""
    print("\n" + "="*70)
    print("PAPER SETTINGS REFERENCE (DiffAttack - Appendix F)")
    print("="*70)
    print("\nAll I-FGSM-based methods:")
    print("  • Steps: 10")
    print("  • Maximum perturbation (ε): 16 (on 0-255 scale) = 16/255 ≈ 0.0627")
    print("  • Step size: 1.6 (on 0-255 scale) = 1.6/255 ≈ 0.00627")
    print("\nMethod-specific parameters:")
    print("  MI-FGSM:")
    print("    • Decay factor: 1.0")
    print("\n  DI-FGSM:")
    print("    • Transformation probability: 0.5")
    print("\n  TI-FGSM:")
    print("    • Kernel size: 7")
    print("\n  PI-FGSM:")
    print("    • Amplification factor: 10")
    print("\n  S2I-FGSM:")
    print("    • Inner iterations: 20")
    print("    • Tuning factor: 0.5")
    print("    • Standard deviation: 16")
    print("="*70 + "\n")

if __name__ == "__main__":
    args = parser.parse_args()
    
    print_paper_settings()
    print(f"Current Configuration:")
    # Echo canonicalized names for clarity
    try:
        att_name = canonicalize_attack_method(args.attack_method)
    except Exception as e:
        print(str(e))
        sys.exit(1)
    model_name_print = canonicalize_model_name(args.model_name)
    print(f"  Attack Method: {att_name}")
    print(f"  Surrogate Model: {model_name_print}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Resolution: {args.res}x{args.res}")
    print(f"\nAttack Parameters:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Epsilon: {args.epsilon:.6f} ({args.epsilon*255:.2f}/255)")
    print(f"  Step Size: {args.step_size:.6f} ({args.step_size*255:.2f}/255)")
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Load labels
    with open(args.label_path, "r") as f:
        label = np.array([int(line.rstrip()) - 1 for line in f.readlines()])
    
    if args.is_test:
        # Test mode
        all_clean_images = natsorted(glob.glob(os.path.join(args.images_root, "*originImage*")), alg=ns.PATH)
        all_adv_images = natsorted(glob.glob(os.path.join(args.images_root, "*adv_image*")), alg=ns.PATH)
        
        images = []
        adv_images = []
        
        for img_path, adv_path in zip(all_clean_images, all_adv_images):
            img = Image.open(img_path).convert('RGB').resize((args.res, args.res))
            images.append(np.array(img).astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0)
            
            adv = Image.open(adv_path).convert('RGB').resize((args.res, args.res))
            adv_images.append(np.array(adv).astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0)
        
        images = np.concatenate(images)
        adv_images = np.concatenate(adv_images)
        
        model_transfer(images, adv_images, label, args.res, save_path=save_dir,
                      fid_path=args.images_root, args=args)
        sys.exit(0)
    
    # Attack mode
    surrogate_model = load_surrogate_model(args.model_name)
    
    all_images = natsorted(glob.glob(os.path.join(args.images_root, "*")), alg=ns.PATH)
    
    images = []
    adv_images = []
    clean_all_acc = 0
    adv_all_acc = 0
    
    for ind, image_path in enumerate(all_images):
        print(f"\nProcessing image {ind+1}/{len(all_images)}: {os.path.basename(image_path)}")
        
        # Load and save original image
        tmp_image = Image.open(image_path).convert('RGB')
        tmp_image.save(os.path.join(save_dir, str(ind).rjust(4, '0') + "_originImage.png"))
        
        # Create save directory for this image
        img_save_dir = os.path.join(save_dir, str(ind).rjust(4, '0'))
        os.makedirs(img_save_dir, exist_ok=True)
        
        # Run attack
        adv_image, clean_acc, adv_acc = run_baseline_attack(
            tmp_image, label[ind:ind + 1], surrogate_model, args
        )
        
        # Save adversarial image
        Image.fromarray(adv_image).save(os.path.join(img_save_dir, "adv_image.png"))
        
        # Store for batch processing
        adv_images.append((adv_image.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2))
        
        tmp_image = tmp_image.resize((args.res, args.res), resample=Image.LANCZOS)
        images.append((np.array(tmp_image).astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2))
        
        clean_all_acc += clean_acc
        adv_all_acc += adv_acc
        
        print(f"  Clean Acc: {clean_acc*100:.2f}% | Adv Acc: {adv_acc*100:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"  Overall Results:")
    print(f"  Clean Accuracy: {clean_all_acc / len(all_images) * 100:.2f}%")
    print(f"  Adversarial Accuracy: {adv_all_acc / len(all_images) * 100:.2f}%")
    print(f"  Attack Success Rate: {(1 - adv_all_acc / len(all_images)) * 100:.2f}%")
    print(f"{'='*60}\n")
    
    images = np.concatenate(images)
    adv_images = np.concatenate(adv_images)
    
    # Test transferability
    model_transfer(images, adv_images, label, args.res, save_path=save_dir, args=args)