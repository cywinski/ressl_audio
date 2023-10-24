import torch
from dataset.data import *
from network.head import *
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from util.meter import *
import time
from util.torch_dist_sum import *
from util.dist_init import dist_init
import argparse
from network.base_model import *
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--s', type=str, default='cos')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()
print(args)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = torch.flatten(correct[:k]).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size).item())
        return res



def main():
    wandb.init(
    project="ressl-cv",
    config=args,
)

    if args.s == 'cos':
        lr = 0.2
    else:
        lr = 30

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='data', download=True, transform=get_train_augment('cifar10'))
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=get_test_augment('cifar10'))
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='data', download=True, transform=get_train_augment('cifar100'))
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=get_test_augment('cifar100'))
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        train_dataset = TinyImagenet(root='data/tiny-imagenet-200/train', transform=get_train_augment('tinyimagenet'))
        test_dataset = TinyImagenet(root='data/tiny-imagenet-200/val', transform=get_test_augment('tinyimagenet'))
        num_classes = 200
    else:
        train_dataset = datasets.STL10(root='data', download=True, split='train', transform=get_train_augment('stl10'))
        test_dataset = datasets.STL10(root='data', download=True, split='test', transform=get_test_augment('stl10'))
        num_classes = 10

    dim_in = 512
    pre_train = ModelBase(dataset=args.dataset)

    prefix = 'net.'

    state_dict = torch.load('checkpoints/' + args.checkpoint, map_location='cpu')['model']
    for k in list(state_dict.keys()):
        if not k.startswith(prefix):
            del state_dict[k]
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict[k]
            del state_dict[k]
    pre_train.load_state_dict(state_dict)

    model = LinearHead(pre_train, dim_in=dim_in, num_class=num_classes)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)

    torch.backends.cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    if args.s == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * train_loader.__len__())
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    best_acc = 0
    best_acc5 = 0

    for epoch in range(args.epochs):
        # ---------------------- Train --------------------------

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            train_loader.__len__(),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch)
        )
        end = time.time()

        model.eval()
        for i, (image, label) in enumerate(train_loader):
            data_time.update(time.time() - end)

            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            out = model(image)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item())
            wandb.log({"train/loss_linear": loss.item()})

            if i % 10 == 0:
                progress.display(i)

            if args.s == 'cos':
                scheduler.step()

        if args.s == 'step':
            scheduler.step()

        # ---------------------- Test --------------------------
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            end = time.time()
            for i, (image, label) in enumerate(test_loader):

                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                # compute output
                output = model(image)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1, image.size(0))
                top5.update(acc5, image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


        top1_acc = top1.avg
        top5_acc = top5.avg

        best_acc = max(top1_acc, best_acc)
        best_acc5 = max(top5_acc, best_acc5)

        wandb.log({
            "test/Acc@1": top1_acc,
            "test/Acc@5": top5_acc,
            "test/Best_Acc@1": best_acc,
            "test/Best_Acc@5": best_acc5
        })

        print('Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(epoch, top1_acc=top1_acc, top5_acc=top5_acc, best_acc=best_acc, best_acc5=best_acc5))

if __name__ == "__main__":
    main()
