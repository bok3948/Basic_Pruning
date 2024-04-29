import torch
import torch.nn as nn   

def load_pruned_model(ori_model, args):
    ori_model = ori_model.cpu()
    checkpoint = torch.load(args.resume, map_location='cpu')
    save_size = checkpoint['save_size']

    model = make_pruned_model(ori_model, "", save_size)

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"resume weights {msg}")
    return model, checkpoint


def make_pruned_model(module, parent_name="", save_size={}):
    for name, child in module.named_children():
        layer_full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(child, nn.Conv2d):
            new_conv = torch.nn.Conv2d(
                                in_channels=save_size[layer_full_name][1],
                                out_channels=save_size[layer_full_name][0],
                                kernel_size=child.kernel_size,
                                stride=child.stride, 
                                padding=child.padding, 
                                dilation=child.dilation, 
                                groups=child.groups, 
                                bias=child.bias is not None, 
                                padding_mode=child.padding_mode
                                )
            setattr(module, name, new_conv)

        elif isinstance(child, nn.BatchNorm2d):
            new_norm = torch.nn.BatchNorm2d(
                                    num_features=save_size[layer_full_name][0],
                                    eps=child.eps,
                                    momentum=child.momentum,
                                    affine=child.affine,
                                    track_running_stats=child.track_running_stats)
            setattr(module, name, new_norm)

        elif isinstance(child, nn.Linear):
            new_linear = torch.nn.Linear(
                    in_features=save_size[layer_full_name][1],
                    out_features=child.out_features,
                    bias=child.bias is not None
            )       
            setattr(module, name, new_linear)

        else:
            make_pruned_model(child, layer_full_name, save_size)

    return module

