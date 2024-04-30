"""
Global Magnitude pruning for CNN.
(1) granularity: filter wise pruning
(2) importance criterion : mean of L1 norm in filter
(3) how to choose per layer pruning ratio: adaptively 
(4) pruning schedule: iterative pruning
(5) pruning scope : Global pruning
"""
import os
import json
import copy
import argparse
import time
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn

from timm import utils
from timm import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from torch.utils.tensorboard import SummaryWriter

from model import VGG

from util.datasets import build_dataset
from util.profiler import profiler, get_layer_size
from util import misc as misc
from util.prepare import prepare_model
from util.load import load_pruned_model
from engine import train_one_epoch, evaluate
from prune import get_layer, global_prune, get_del_indexes, reset_in_channel_index
from get_res import get_residual, res_get_del_indexes

def get_args_parser():
    parser = argparse.ArgumentParser(description='Magnitude pruning', add_help=False)

    #setting
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--device', default='cuda', help='cpu vs cuda')

    #data load
    parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'CIFAR10'])
    parser.add_argument('--data_path', default='/mnt/d/data/image', type=str, help='path to data')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')

    #model
    parser.add_argument('--model', default='vgg16', type=str, help='model name',
                        choices=['vgg16', 'resnet18','resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--pretrained', default='./resnet18_best_checkpoint.pth', help='get pretrained weights from checkpoint')

    #optimizer
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)

    # Learning rate schedule parameters
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=1, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    #run
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    #save and log
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save scale, empty for no saving')

    #pruning parameters
    parser.add_argument('--start_iter', default=0, type=int, help='resume from iteration')
    parser.add_argument('--pruning_ratio', default=0.7, type=float, help='pruning ratio')
    parser.add_argument('--per_iter_pruning_ratio', default=0.05, type=float, help='filters * per_iter_pruning_ratio = filters to be pruned per iter')
    parser.add_argument('--min_ratio', default=0.01, type=float, help='minimum ratio of filters to be pruned')

    #distillation
    parser.add_argument('--do_KD', action='store_true', default=False,
                    help='do distillation')
    parser.add_argument('--KD_loss_weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=2.0)

    return parser

def main(args):

    if args.distributed:
        misc.init_distributed_mode(args)
    device = torch.device(args.device) 
    torch.cuda.empty_cache()

    #load validation dataset for evaluation
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if args.output_dir:
        output_dir = f"{args.output_dir}/{args.model}"

        log_dir = output_dir +"/" + "log"
        log_writer = SummaryWriter(log_dir=log_dir)

        with open(f"{output_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.distributed:
        #Sampler
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, 
        num_workers=3,
        drop_last=True, sampler=train_sampler
    )
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=10, 
        num_workers=3,
        drop_last=False
    )

    # load model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
    ) 

    if args.pretrained:
        if args.pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')

        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        ori_test_stats = {"acc1": checkpoint['max_accuracy']} if 'max_accuracy' in checkpoint else None

    ori_model = copy.deepcopy(model)

    if ori_test_stats is None:
        ori_test_stats = evaluate(data_loader_val, ori_model.to(device), device, args)
    print(f"Original model test : {ori_test_stats}")

    full_layers = get_layer(model, parent_name="", pruning_layers=None, include_bn=False)
    pruning_layers = []
    for i, k in enumerate(full_layers):
        if ("classifier" not in k) and ("fc" not in k):
            pruning_layers.append(k)


    print(f" pruning layers: {pruning_layers}")

    # optimizer         
    optimizer = create_optimizer(args, model)

    # lr_scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)

    #loss
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    best_model, current_num_para = None, torch.inf

    ori_num_para = sum(p.numel() for p in ori_model.parameters() if p.requires_grad)

    filter_num = {}
    for pr in full_layers:
        filter_num[pr] = get_layer_size(ori_model, pr)[0]
    exclude = ["conv1"]

    start_time = time.time()
    iter = args.start_iter
    result_save = {"remain_weights(%)": [100], "acc1": [ori_test_stats["acc1"]]}
    while ((1 - args.pruning_ratio) * ori_num_para) < current_num_para:
        if best_model is not None:
            model = best_model
            result_save["remain_weights(%)"].append((current_num_para/ori_num_para)*100)
            result_save["acc1"].append(max_accuracy)

        print(f"-"*50 + f"iteration: [{iter}]" + "-"*50)

        max_accuracy = 0.0

        print(f"-"*50 + "start pruning" + "-"*50)
        for k in exclude:
            if k in pruning_layers:
                pruning_layers.remove(k)

        #model to cpu
        try:
            if args.distributed:
                model = model.module
        except:
            pass
        model = model.cpu()
        if "resnet" in args.model:
            stages = get_residual(pruning_layers)
            results, exclude = res_get_del_indexes(model, pruning_layers, filter_num, stages, args)

            residuals = []
            stages1 = get_residual(full_layers[:-1])
            for stage in stages1.keys():
                for l in stages1[stage]:
                    residuals.append(l)
        else:
            results, exclude = get_del_indexes(model, pruning_layers, filter_num, args)
            residuals = []
        
        reset_in_channel_index()
        model = global_prune(model, results, "", residuals)
        
        current_num_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Current pruned model #parameters {(current_num_para/ori_num_para)*100:.2f}%")
        
        for k in full_layers:
            print(f"{k} : {(get_layer_size(model, k)[0] / filter_num[k]) * 100 :.2f} #remain weights(%) ")

        save_size = {}
        tem = get_layer(ori_model, parent_name="", pruning_layers=None, include_bn=True)
        for k in tem:
            save_size[k] = get_layer_size(model, k)
        del tem

        if args.resume:
            print(f"-"*50 + "Resuming" + "-"*50)
            print(f"building pruned model using checkpoint['save_size']")
            model, checkpoint = load_pruned_model(ori_model, args)

        model, model_without_ddp = prepare_model(model, args)
        if args.do_KD:
            teacher = copy.deepcopy(ori_model)
            teacher, ori_model_without_ddp = prepare_model(teacher, args)
        else:
            teacher = None

        del optimizer
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        loss_scaler = torch.cuda.amp.GradScaler()

        #resume
        if args.resume:
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

        # val_stats = evaluate(data_loader_val, model, device, args)

        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(data_loader_train, model, 
                optimizer, criterion, loss_scaler, None, device,
                epoch, None, teacher, args
            )

            if lr_scheduler is not None:
                lr_scheduler.step(epoch)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        print("Distributing BatchNorm running means and vars")
                    utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            
            val_stats = evaluate(data_loader_val, model, device, args)

            if log_writer is not None:
                log_writer.add_scalar('perf/acc1', val_stats['acc1'], epoch)

            if max_accuracy < val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
                best_model = model
                if args.output_dir:
                    checkpoint_path = output_dir + "/" +  f'{iter}_best_checkpoint.pth'
                    misc.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            "max_accuracy": max_accuracy,
                            "save_size": save_size,
                        }, checkpoint_path)

            #Save
            if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 ==args.epochs):
                misc.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    "save_size": save_size,

                }, os.path.join(output_dir, f'{iter}_checkpoint_{epoch}.pth')
                )

            log_stats = {**{f'iter:{iter}_val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch, 'best_acc': max_accuracy, 'remain_weights(%)': (current_num_para/ori_num_para)*100}
            
            if misc.is_main_process():
                with open(f"{output_dir}/pruning_val_log.txt", mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats) + '\n')

        iter += 1

 
    with open(f"{output_dir}/result_save.txt", mode='a', encoding='utf-8') as f:
        f.write(json.dumps(result_save) + '\n')
    
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    log_stats["total_time"] = total_time_str
    print(f'Training time {total_time_str}')

    #get ori model test stats
    pf = profiler(dummy_size=(1, 3, args.input_size, args.input_size))
    ori_test_stats["name"] = f"pytorch_{args.model}"
    ori_test_stats.update(pf.summary(ori_model, "cpu"))

    prune_test_stats = {"acc1": max_accuracy}
    prune_test_stats["name"] = f"pytorch_pruned_{args.model}"
    prune_test_stats.update(pf.summary(best_model, "cpu"))

    print(f"Original model test stats: {ori_test_stats}")
    print(f"Pruned model test stats: {prune_test_stats}")

    #acc change
    acc_change = (prune_test_stats["acc1"] - ori_test_stats["acc1"])
    print(f"Acc change(acc(prune) - acc(ori): {acc_change:.2f} %")

    summary = {"ori model": ori_test_stats, "prune model": prune_test_stats}

    with open(f"{output_dir}/pruning_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)