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
)
import nnAudio.features

parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", type=str, default="data/audioset/all/")
parser.add_argument("--dataset", type=str, default="audioset")
parser.add_argument("--port", type=int, default=23456)
parser.add_argument("--k", type=int, default=4096)
parser.add_argument("--m", type=float, default=0.99)
parser.add_argument("--weak", default=False, action="store_true")
parser.add_argument("--epochs", type=int, default=800)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--base_lr", type=float, default=0.06)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--run_name", type=str, default="run")
parser.add_argument("--prenormalize", type=bool, default=False)
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


def calc_norm_stats(data_loader, n_stats=10000, device="cuda"):
    # Calculate normalization statistics from the training dataset (spectrograms).
    n_stats = min(n_stats, len(data_loader.dataset))
    print(
        f"Calculating mean/std using random {n_stats} samples from population {len(data_loader.dataset)} samples..."
    )
    X = []
    for wavs in data_loader:
        for wav in wavs:
            lms_batch = (to_spec(wav) + torch.finfo().eps).log().unsqueeze(1)
            X.extend([x for x in lms_batch.detach().cpu().numpy()])
            if len(X) >= n_stats:
                break
    X = np.stack(X)
    norm_stats = np.array([X.mean(), X.std()])
    print(f"  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}")
    return norm_stats


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
    for i, (wav1, wav2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        wandb.log({"lr": optimizer.param_groups[0]["lr"]})
        data_time.update(time.time() - end)

        # Raw audio to log-mel spectrograms
        img1 = (to_spec(wav1) + torch.finfo().eps).log().unsqueeze(1)
        img2 = (to_spec(wav2) + torch.finfo().eps).log().unsqueeze(1)

        if pre_norm is not None:
            img1 = pre_norm(img1)
            img2 = pre_norm(img2)

        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)

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

        progress.display(i)


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

    model = ReSSL(K=args.k, m=args.m)
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

    checkpoint_path = "checkpoints/ressl-{}-epochs-{}-bs-{}-{}.pth".format(
        args.dataset, args.epochs, args.batch_size, args.run_name
    )
    print("checkpoint_path:", checkpoint_path)
    if os.path.exists(checkpoint_path) and args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print(checkpoint_path, "found, start from epoch", start_epoch)
    else:
        start_epoch = 0
        print(checkpoint_path, "not found, start from epoch 0")

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
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            checkpoint_path,
        )


if __name__ == "__main__":
    main()
