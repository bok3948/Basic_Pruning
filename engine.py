import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

from timm.utils import accuracy
import util.misc as misc


def get_loss_one_logit(student_logit, teacher_logit, temperature=2.0):
    t = temperature # distillation temperature
    return F.kl_div(
        input=F.log_softmax(student_logit / t, dim=-1),
        target=F.softmax(teacher_logit / t, dim=-1),
        reduction="batchmean"
    ) * (t ** 2)

def train_one_epoch(data_loader: Iterable, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, loss_scaler,
                lr_scheduler=None, 
                device=None, epoch=None,
                log_writer=None, teacher=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch [{}]'.format(epoch)
    
    for it, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(True):
            logits = model(inputs)
        loss = criterion(logits, labels)

        # use distillation
        if args.do_KD:
            with torch.no_grad():
                teacher_output = teacher(inputs)
            distillation_loss = args.KD_loss_weight * get_loss_one_logit(logits, teacher_output, args.temperature)
            loss += distillation_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
    #gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def evaluate(data_loader, model, device, args):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="\t")
    header = 'Test:'

    for inputs, labels in metric_logger.log_every(data_loader, args.print_freq, header):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if args.device == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(inputs)
        else:
            logits = model(inputs)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = inputs.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}