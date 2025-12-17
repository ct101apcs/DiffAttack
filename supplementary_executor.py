#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from typing import List


def canon_attack(name: str) -> str:
    t = name.strip().lower().replace('_', '-')
    mapping = {
        'fgsm': 'FGSM',
        'mi-fgsm': 'MI-FGSM', 'mifgsm': 'MI-FGSM', 'mi': 'MI-FGSM',
        'di-fgsm': 'DI-FGSM', 'difgsm': 'DI-FGSM', 'di': 'DI-FGSM',
        'ti-fgsm': 'TI-FGSM', 'tifgsm': 'TI-FGSM', 'ti': 'TI-FGSM',
        'pi-fgsm': 'PI-FGSM', 'pifgsm': 'PI-FGSM', 'pi': 'PI-FGSM',
        's2i-fgsm': 'S2I-FGSM', 's2ifgsm': 'S2I-FGSM', 's2i': 'S2I-FGSM',
    }
    return mapping.get(t, name)


def canon_model(name: str) -> str:
    n = name.strip().lower().replace(' ', '').replace('_', '-')
    aliases = {
        'resnet50': {'resnet', 'resnet50', 'rn50', 'resnet-50', 'restnet50'},
        'vgg19': {'vgg', 'vgg19', 'vgg-19'},
        'vit_b_16': {'vit', 'vitb16', 'vit-b16', 'vit-b-16', 'vit-b/16', 'vit_b_16'},
        'swin_b': {'swin', 'swinb', 'swin-b', 'swin_b'},
        'inception_v3': {'inception', 'inceptionv3', 'inception-v3', 'inception_v3'},
        'mobilenet_v2': {'mobilenet', 'mobilenetv2', 'mobilenet-v2', 'mobilenet_v2'},
    }
    for k, opts in aliases.items():
        if n in opts:
            return k
    return name


def parse_list(arg: str, defaults: List[str]) -> List[str]:
    if arg.lower() in {'all', 'auto'}:
        return defaults
    items = [x.strip() for x in arg.split(',') if x.strip()]
    return items or defaults


def main():
    parser = argparse.ArgumentParser(description='Run supplementary.py across multiple models/attacks')
    parser.add_argument('--save_root', default='supplementary_runs', type=str,
                        help='Base directory to store results (subfolders per model)')
    parser.add_argument('--images_root', default='datasets/imagenet-compatible/images', type=str,
                        help='Dataset images root')
    parser.add_argument('--label_path', default='datasets/imagenet-compatible/labels.txt', type=str,
                        help='Labels file path')
    parser.add_argument('--res', default=224, type=int)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--epsilon', default=16/255, type=float)
    parser.add_argument('--step_size', default=1.6/255, type=float)
    parser.add_argument('--decay', default=1.0, type=float)
    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--kernel_size', default=7, type=int)
    parser.add_argument('--amplification', default=10, type=int)

    parser.add_argument('--models', default='resnet50,vgg19,vit_b_16,swin_b,inception_v3,mobilenet_v2', type=str,
                        help='Comma-separated models or "all"')
    parser.add_argument('--attacks', default='FGSM,MI-FGSM,DI-FGSM,TI-FGSM,PI-FGSM', type=str,
                        help='Comma-separated attacks or "all"')

    parser.add_argument('--dry_run', action='store_true', help='Print commands without executing')

    args = parser.parse_args()

    default_models = ['resnet50', 'vgg19', 'vit_b_16', 'swin_b', 'inception_v3', 'mobilenet_v2']
    default_attacks = ['FGSM', 'MI-FGSM', 'DI-FGSM', 'TI-FGSM', 'PI-FGSM']

    models = [canon_model(m) for m in parse_list(args.models, default_models)]
    attacks = [canon_attack(a) for a in parse_list(args.attacks, default_attacks)]

    os.makedirs(args.save_root, exist_ok=True)

    py = sys.executable
    failures = []

    for m in models:
        # Group both MI/DI in a single invocation is not supported by supplementary.py,
        # so invoke once per attack for clarity/logging.
        for a in attacks:
            save_dir = os.path.join(args.save_root, f"{m}", a.replace('/', '-'))
            os.makedirs(save_dir, exist_ok=True)

            cmd = [
                py, 'supplementary.py',
                '--attack_method', a,
                '--model_name', m,
                '--save_dir', save_dir,
                '--images_root', args.images_root,
                '--label_path', args.label_path,
                '--res', str(args.res),
                '--iterations', str(args.iterations),
                '--epsilon', str(args.epsilon),
                '--step_size', str(args.step_size),
                '--decay', str(args.decay),
                '--prob', str(args.prob),
                '--kernel_size', str(args.kernel_size),
                '--amplification', str(args.amplification),
            ]

            print('Running:', ' '.join(cmd))
            if args.dry_run:
                continue
            try:
                rc = subprocess.run(cmd, check=False)
                if rc.returncode != 0:
                    failures.append((m, a, rc.returncode))
            except Exception as e:
                print('Error:', e)
                failures.append((m, a, str(e)))

    if failures:
        print('\nSome runs failed:')
        for m, a, err in failures:
            print(f' - model={m}, attack={a}, error={err}')
        sys.exit(1)
    else:
        print('\nAll runs completed successfully.')


if __name__ == '__main__':
    main()
