# --------------------------------------------------------
# ImageNet-21K Pretraining for The Masses
# Copyright 2021 Alibaba MIIL (c)
# Licensed under MIT License [see LICENSE file for details]
# Written by Tal Ridnik
# --------------------------------------------------------

import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler
import torch.multiprocessing as mp

from src_files.data_loading.data_loader import create_data_loaders
from src_files.helper_functions.distributed import print_at_master, setup_distrib, reduce_tensor, num_distrib
from src_files.helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings
from src_files.models import create_model
from src_files.loss_functions.losses import CrossEntropyLS
from torch.cuda.amp import GradScaler, autocast
from src_files.optimizers.create_optimizer import create_optimizer

parser = argparse.ArgumentParser(description='PyTorch ImageNet21K Single-label Training')
parser.add_argument('--data_path', type=str)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--model_name', default='tresnet_m')
parser.add_argument('--model_path', default='./tresnet_m.pth', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--num_classes', default=11221, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument("--label_smooth", default=0.2, type=float)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')



def main():
    args = parser.parse_args()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function

        # processes = []
        # for rank in range(ngpus_per_node):
        #     p = mp.Process(target=main_worker, args=(rank, ngpus_per_node, args))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)



    pass


def main_worker(gpu ,ngpu_per_node, args):
    torch.cuda.set_device(gpu)
    args.local_rank = gpu
    args.rank  = args.rank * ngpu_per_node + args.local_rank
    # arguments
    # args = parser.parse_args()

    # EXIF warning silent
    silence_PIL_warnings()

    # Setup model
    model = create_model(args).cuda(gpu)

    # create optimizer
    optimizer = create_optimizer(model, args)

    # setup distributed
    model = setup_distrib(model, args)

    # Data loading
    train_loader, val_loader = create_data_loaders(args)

    # Actuall Training
    train_21k(model, train_loader, val_loader, optimizer, args)


def train_21k(model, train_loader, val_loader, optimizer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    # set loss
    loss_fn = CrossEntropyLS(args.label_smooth)

    # set scheduler
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.1, cycle_momentum=False, div_factor=20)

    # set scalaer
    scaler = GradScaler()

    # training loop
    for epoch in range(args.epochs):
        progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train epoch
        # print_at_master("\nEpoch {}".format(epoch))
        epoch_start_time = time.time()
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            with autocast():  # mixed precision
                output = model(input)
                loss = loss_fn(output, target) # note - loss also in fp16
            losses.update(loss.item(), input.size(0))
            model.zero_grad()
            scheduler.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # batch_time = time.time() - batch_start_time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 20 == 0:
                progress.display(i)

        epoch_time = time.time() - epoch_start_time
        print_at_master(
            "\nFinished Epoch, Training Rate: {:.1f} [img/sec]".format(len(train_loader) *
                                                                      args.batch_size / epoch_time * max(num_distrib(),
                                                                                                         1)))

        # validation epoch
        validate_21k(val_loader, model)


def validate_21k(val_loader, model):
    print_at_master("starting validation")
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # mixed precision
            with autocast():
                logits = model(input).float()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            if args.distributed:
                acc1 = reduce_tensor(acc1, num_distrib())
                acc5 = reduce_tensor(acc5, num_distrib())
                torch.cuda.synchronize()
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

    print_at_master("Validation results:")
    print_at_master('Acc_Top1 [%] {:.2f},  Acc_Top5 [%] {:.2f} '.format(top1.avg, top5.avg))
    model.train()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_at_master('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == '__main__':
    main()
