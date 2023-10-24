import torch
from util.torch_dist_sum import *
from util.meter import *
from network.ressl import ReSSL
import time
import os
from dataset.data import *
import math
import argparse
from torch.utils.data import DataLoader
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--k', type=int, default=4096)
parser.add_argument('--m', type=float, default=0.99)
parser.add_argument('--weak', default=False, action='store_true')
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--base_lr', type=float, default=0.06)
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
        param_group['lr'] = lr


def train(train_loader, model, optimizer, epoch, iteration_per_epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    ce_losses = AverageMeter('CE', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ce_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img1, img2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        wandb.log({'lr': optimizer.param_groups[0]['lr']})
        data_time.update(time.time() - end)
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)

        # compute output
        loss = model(img1, img2)
        wandb.log({'loss': loss.item()})

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


def main():
    wandb.init(
    project="ressl-cv",
    config=args,
)

    model = ReSSL(dataset=args.dataset, K=args.k, m=args.m)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'cifar10':
        dataset = CIFAR10Pair(root='data', download=True, transform=get_contrastive_augment('cifar10'), weak_aug=get_weak_augment('cifar10'))
    elif args.dataset == 'stl10':
        dataset = STL10Pair(root='data', download=True, split='train+unlabeled', transform=get_contrastive_augment('stl10'), weak_aug=get_weak_augment('stl10'))
    elif args.dataset == 'tinyimagenet':
        dataset = TinyImagenetPair(root='data/tiny-imagenet-200/train', transform=get_contrastive_augment('tinyimagenet'), weak_aug=get_weak_augment('tinyimagenet'))
    else:
        dataset = CIFAR100Pair(root='data', download=True, transform=get_contrastive_augment('cifar100'), weak_aug=get_weak_augment('cifar100'))

    train_loader = DataLoader(dataset, shuffle=True, num_workers=6, pin_memory=True, batch_size=args.batch_size, drop_last=True)
    iteration_per_epoch = train_loader.__len__()

    checkpoint_path = 'checkpoints/ressl-{}.pth'.format(args.dataset)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(checkpoint_path, 'found, start from epoch', start_epoch)
    else:
        start_epoch = 0
        print(checkpoint_path, 'not found, start from epoch 0')


    model.train()
    for epoch in range(start_epoch, epochs):
        train(train_loader, model, optimizer, epoch, iteration_per_epoch, args.base_lr)
        torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }, checkpoint_path)


if __name__ == "__main__":
    main()
