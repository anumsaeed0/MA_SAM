import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module
from sam_fact_tt_image_encoder import Fact_tt_Sam
from segment_anything import sam_model_registry
from trainer import trainer_run
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/weka/wekafs/rad-megtron/cchen/synapseCT/Training/2D_all_5slice')
parser.add_argument('--output', type=str, default='/mnt/weka/wekafs/rad-megtron/cchen/project_results/MA_SAM/results-1')
parser.add_argument('--num_classes', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--n_gpu', type=int, default=1)  # safer default
parser.add_argument('--base_lr', type=float, default=0.0008)

parser.add_argument('--max_epochs', type=int, default=400)
parser.add_argument('--stop_epoch', type=int, default=300)

parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vit_name', type=str, default='vit_h')
parser.add_argument('--ckpt', type=str, default='/mnt/weka/wekafs/rad-megtron/cchen/PretrainedModel/sam_vit_h_4b8939.pth')
parser.add_argument('--adapt_ckpt', type=str, default=None)
parser.add_argument('--rank', type=int, default=32)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--warmup_period', type=int, default=250)
parser.add_argument('--AdamW', action='store_true')
parser.add_argument('--module', type=str, default='sam_fact_tt_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--lr_exp', type=float, default=7)

# acceleration
parser.add_argument('--tf32', action='store_true')
parser.add_argument('--compile', action='store_true')
parser.add_argument('--use_amp', action='store_true')
parser.add_argument('--skip_hard', action='store_true')

args = parser.parse_args()

# Enable default acceleration options if desired
args.warmup = True
args.AdamW = True
args.tf32 = True
args.compile = False
args.use_amp = True
args.skip_hard = True


if __name__ == "__main__":

    # ---------------------------
    # Device selection (CPU/GPU)
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        args.n_gpu = torch.cuda.device_count()
    else:
        args.n_gpu = 0
        args.use_amp = False  # AMP only valid on CUDA

    # ---------------------------
    # TF32 (CUDA only)
    # ---------------------------
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------
    # Deterministic settings
    # ---------------------------
    if device.type == "cuda":
        if not args.deterministic:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True

    # ---------------------------
    # Seed
    # ---------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------
    # Output folder
    # ---------------------------
    os.makedirs(args.output, exist_ok=True)

    # ---------------------------
    # Load SAM
    # ---------------------------
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0., 0., 0.],
        pixel_std=[1., 1., 1.]
    )

    # ---------------------------
    # Register model
    # ---------------------------
    pkg = import_module(args.module)
    net = pkg.Fact_tt_Sam(sam, args.rank, s=args.scale)

    # Move to device (CPU or GPU)
    net = net.to(device)

    # Optional torch.compile (PyTorch 2.x)
    if args.compile:
        net = torch.compile(net)

    # Load finetuned checkpoint
    if args.adapt_ckpt is not None:
        net.load_parameters(args.adapt_ckpt)

    # Multi-mask logic
    multimask_output = args.num_classes > 1
    low_res = img_embedding_size * 4

    # ---------------------------
    # Save config
    # ---------------------------
    config_file = os.path.join(args.output, 'config.txt')
    with open(config_file, 'w') as f:
        for key, value in args.__dict__.items():
            f.write(f'{key}: {value}\n')

    # ---------------------------
    # Train
    # ---------------------------
    trainer_run(args, net, args.output, multimask_output, low_res, device)
