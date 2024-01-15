import torch
from util.torch_dist_sum import *
from util.meter import *
from network.ressl import ReSSL
import time
import os
import math
import argparse
from torch.utils.data import DataLoader
import wandb
import random
import numpy as np
from pathlib import Path
from dataset.audio_data import WavDatasetPair

from dataset.audio_augmentations import (
    get_contrastive_augment,
    get_weak_augment,
    PrecomputedNorm,
    NormalizeBatch,
)
import nnAudio.features
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", type=str, default="data/train_wav_16k/")
parser.add_argument("--dataset", type=str, default="audioset")
parser.add_argument("--port", type=int, default=23456)
parser.add_argument("--k", type=int, default=4096)
parser.add_argument("--m", type=float, default=0.99)
parser.add_argument("--weak", default=False, action="store_true")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--base_lr", type=float, default=0.001)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--run_name", type=str, default="run")
parser.add_argument("--prenormalize", type=bool, default=False)
parser.add_argument("--postnormalize", type=bool, default=False)
parser.add_argument("--log_interval", type=int, default=60)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--tau_s", type=float, default=0.1)
parser.add_argument("--tau_t", type=float, default=0.04)
args = parser.parse_args()
print(args)

epochs = args.epochs
warm_up = 5


def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


to_spec = nnAudio.features.MelSpectrogram(
    sr=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=160,
    n_mels=64,
    fmin=60,
    fmax=7800,
    center=True,
    power=2,
    verbose=False,
)

post_norm = NormalizeBatch()


def train(
    train_loader,
    model,
    optimizer,
    epoch,
    iteration_per_epoch,
    base_lr,
    pre_norm,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    ce_losses = AverageMeter("CE", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ce_losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    to_spec.to("cuda")
    for i, (wav1, wav2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"]},
        )
        data_time.update(time.time() - end)

        wav1 = wav1.cuda(non_blocking=True)
        wav2 = wav2.cuda(non_blocking=True)

        # Raw augmented audio to log-mel spectrograms
        img1 = (to_spec(wav1) + torch.finfo().eps).log().unsqueeze(1)
        img2 = (to_spec(wav2) + torch.finfo().eps).log().unsqueeze(1)

        if args.postnormalize:
            img1 = post_norm(img1)
            img2 = post_norm(img2)

        # compute output
        loss = model(img1, img2)
        wandb.log({"loss": loss.item()})

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        ce_losses.update(loss.item(), img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0:
            progress.display(i)
            wandb.log(
                {
                    "wav_strong_aug": wandb.Audio(wav1[0].cpu(), sample_rate=16000),
                    "wav_weak_aug": wandb.Audio(wav2[0].cpu(), sample_rate=16000),
                    "img_strong_aug": wandb.Image(img1),
                    "img_weak_aug": wandb.Image(img2),
                },
            )


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def main():
    wandb.init(project="ressl-audio", config=args, name=args.run_name)
    generator = seed_everything()

    model = ReSSL(K=args.k, m=args.m, tau_s=args.tau_s, tau_t=args.tau_t)
    model = model.cuda()
    print(model)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4
    )

    files = sorted(Path(args.audio_dir).glob("*.wav"))

    if args.prenormalize:
        # NOTE: No augmentations applied!
        dataset_no_aug = WavDatasetPair(
            sample_rate=16000,
            audio_files=files,
            labels=None,
            random_crop=True,
            contrastive_aug=None,
            weak_aug=None,
        )
        train_loader_no_aug = DataLoader(
            dataset_no_aug,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            batch_size=args.batch_size,
            drop_last=True,
            generator=generator,
        )
        pre_norm = PrecomputedNorm(calc_norm_stats(train_loader_no_aug))
    else:
        pre_norm = None

    dataset = WavDatasetPair(
        sample_rate=16000,
        audio_files=files,
        labels=None,
        random_crop=True,
        contrastive_aug=get_contrastive_augment(),
        weak_aug=get_weak_augment(),
    )
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        batch_size=args.batch_size,
        drop_last=True,
        generator=generator,
    )
    iteration_per_epoch = train_loader.__len__()

    checkpoint_dir = "checkpoints/ressl-{}-epochs-{}-bs-{}-{}/".format(
        args.dataset, args.epochs, args.batch_size, args.run_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("checkpoint_dir:", checkpoint_dir)
    if os.path.exists(checkpoint_dir) and args.resume:
        checkpoint_path = sorted(glob.glob(checkpoint_dir, "*.pth"))[-1]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print(checkpoint_path, "found, start from epoch", start_epoch)
    else:
        start_epoch = 0
        print("start from epoch 0")

    model.train()
    for epoch in range(start_epoch, epochs):
        train(
            train_loader,
            model,
            optimizer,
            epoch,
            iteration_per_epoch,
            args.base_lr,
            pre_norm,
        )
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                },
                os.path.join(checkpoint_dir, "checkpoint-{}.pth".format(epoch + 1)),
            )

    torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epochs,
                },
                os.path.join(checkpoint_dir, "checkpoint-{}.pth".format(epochs)),
            )


if __name__ == "__main__":
    main()
