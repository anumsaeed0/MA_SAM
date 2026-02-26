import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from icecream import ic
from datetime import datetime


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def worker_init_fn(worker_id):
    import random, numpy as np
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(worker_id)

def trainer_run(args, model, snapshot_path, multimask_output, low_res, device):

    from datasets.dataset import dataset_reader, RandomGenerator

    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs('./training_log', exist_ok=True)

    logging.basicConfig(
        filename='./training_log/' + args.output.split('/')[-1] + '_log.txt',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes

    # -------------------------
    # batch size safety
    # -------------------------
    batch_size = args.batch_size
    if args.n_gpu and args.n_gpu > 1:
        batch_size = args.batch_size * args.n_gpu

    db_train = dataset_reader(
        base_dir=args.root_path,
        split="train",
        num_classes=args.num_classes,
        transform=transforms.Compose([
            RandomGenerator(
                output_size=[args.img_size, args.img_size],
                low_res=[low_res, low_res]
            )
        ])
    )

    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(
        db_train,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # -------------------------
    # MultiGPU support
    # -------------------------
    if args.n_gpu > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    model = model.to(device)
    model.train()

    # Loss functions
    ce_loss = CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(num_classes + 1)

    # -------------------------
    # LR warmup
    # -------------------------
    if args.warmup:
        b_lr = base_lr / max(1, args.warmup_period)
    else:
        b_lr = base_lr

    # -------------------------
    # Optimizer
    # -------------------------
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if args.AdamW:
        optimizer = optim.AdamW(
            parameters,
            lr=b_lr,
            betas=(0.9, 0.999),
            weight_decay=0.1
        )
    else:
        optimizer = optim.SGD(
            parameters,
            lr=b_lr,
            momentum=0.9,
            weight_decay=0.0001
        )

    # -------------------------
    # AMP scaler (GPU only)
    # -------------------------
    if args.use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_enabled = True
    else:
        scaler = None
        autocast_enabled = False

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)

    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    iterator = tqdm(range(max_epoch), ncols=70)

    # ===============================
    # Training Loop
    # ===============================
    for epoch_num in iterator:

        for sampled_batch in trainloader:

            # ---------- load batch ----------
            image_batch = sampled_batch['image']
            label_batch = sampled_batch['label']
            low_res_label_batch = sampled_batch['low_res_label']

            # ---------- reshape logic ----------
            image_batch = image_batch.unsqueeze(2)
            image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)

            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)

            # ---------- device move ----------
            image_batch = image_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            low_res_label_batch = low_res_label_batch.to(device, non_blocking=True)

            # ================= training step =================
            optimizer.zero_grad()

            if autocast_enabled:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(image_batch, multimask_output, args.img_size)

                    loss, loss_ce, loss_dice = calc_loss(
                        outputs,
                        label_batch,
                        ce_loss,
                        dice_loss,
                        args.dice_param
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(image_batch, multimask_output, args.img_size)

                loss, loss_ce, loss_dice = calc_loss(
                    outputs,
                    label_batch,
                    ce_loss,
                    dice_loss,
                    args.dice_param
                )

                loss.backward()
                optimizer.step()

            # ================= LR schedule =================
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
            else:
                if args.warmup:
                    shift_iter = max(0, iter_num - args.warmup_period)
                else:
                    shift_iter = iter_num

                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # ================= Logging =================
            iter_num += 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)

            logging.info(
                f'iteration {iter_num} : loss : {loss.item():.6f}, '
                f'loss_ce: {loss_ce.item():.6f}, '
                f'loss_dice: {loss_dice.item():.6f}'
            )

        # ================= Model Saving =================
        save_interval = 20

        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')

            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)

            logging.info(f"save model to {save_mode_path}")

        if epoch_num >= max_epoch - 1 or epoch_num >= args.stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')

            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)

            logging.info(f"save model to {save_mode_path}")

            iterator.close()
            break

    writer.close()

    return "Training Finished!"
