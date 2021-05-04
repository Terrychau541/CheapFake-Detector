import argparse
import shutil
import time

import numpy as np
import os
from os.path import exists, split, join, splitext

import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ignite.metrics import Accuracy, Precision, Recall

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import drn as models
from drn_seg import DRNSub

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'])
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='drn18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: drn18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--check-freq', default=10, type=int,
                        metavar='N', help='checkpoint frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['linear', 'step'], default='step')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224)
    parser.add_argument('--scale-size', dest='scale_size', type=int, default=256)
    parser.add_argument('--step-ratio', dest='step_ratio', type=float, default=0.1)
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2)
    args = parser.parse_args()
    return args


def main():
    print(' '.join(sys.argv))
    args = parse_args()
    print(args)
    if args.cmd == 'train':
        run_training(args)
    elif args.cmd == 'test':
        test_model(args)


def run_training(args):
    # create model
    model = DRNSub(args.pretrained, num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    best_acc = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Data Augmentation and Conversion to Tensor
    train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    print(train_set.class_to_idx)

    #No data augmentation run when loading validation set
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    #Scheduler reduces lr after 5 epochs without loss reduction in validation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    tloss = []
    vloss = []

    for epoch in range(args.start_epoch, args.epochs):
        
        for param_group in optimizer.param_groups:
            print('Epoch [{}] Learning rate: {}'.format(epoch, param_group['lr']))

        # train for one epoch
        ctloss = train(args, train_loader, model, criterion, optimizer, epoch)
        tloss.append(ctloss)
        

        # evaluate on validation set
        vacc, cvloss = validate(args, val_loader, model, criterion)
        vloss.append(cvloss)
        scheduler.step(cvloss)
        
        
        # remember best acc and save checkpoint
        is_best = vacc > best_acc
        best_acc = max(prec1, best_acc)
        
        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.check_freq == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)
    print("Training Losses:", tloss)
    print("Validation Losses:", vloss)


def test_model(args):
    # create model
    model = DRNSub(args.pretrained, num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #Same as val loader in training
    t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, t),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    validate(args, val_loader, model, criterion)


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = Accuracy()
    prec = Precision()
    recall = Recall()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure metrics and record loss
        acc.update((output, target_var))
        prec.update((output, target_var))
        recall.update((output, target_var))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            atmp = acc.compute()*100
            ptmp = [round(elem*100, 2) for elem in prec.compute().tolist()]
            rtmp = [round(elem*100, 2) for elem in recall.compute().tolist()]
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc:.2f}\t'
                  'Prec {prec}\t'
                  'Recall {rec}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=losses, acc=atmp, prec=ptmp, rec=rtmp))
            
        #update end for next cycle
        end = time.time()
    return losses.avg


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = Accuracy()
    prec = Precision()
    recall = Recall()

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure metrics and record loss
            acc.update((output, target_var))
            prec.update((output, target_var))
            recall.update((output, target_var))
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            

            if i % args.print_freq == 0:
                atmp = acc.compute()*100
                ptmp = [round(elem*100, 2) for elem in prec.compute().tolist()]
                rtmp = [round(elem*100, 2) for elem in recall.compute().tolist()]
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc:.2f}\t'
                      'Prec {prec}'
                      'Recall {rec}'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,acc=atmp, prec=ptmp, rec=rtmp))\
            
            end = time.time()

    # Print overall metrics after running entire validation set
    atmp = acc.compute()*100
    ptmp = [round(elem*100, 2) for elem in prec.compute().tolist()]
    rtmp = [round(elem*100, 2) for elem in recall.compute().tolist()]
    print('Final Results: * Acc {acc:.2f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec {prec}\t'
          'Recall {rec}'.format(acc=atmp, loss=losses, prec=ptmp, rec = rtmp))

    return acc.compute()*100, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


if __name__ == '__main__':
    main()
