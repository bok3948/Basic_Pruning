import torch
import torch.nn as nn

def prepare_model(model, args):

    model = model.to(args.device)
    if args.distributed:
        if args.dist_bn is not None:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    torch.compile(model)
    return model, model_without_ddp