import torch
import torch.nn as nn

def get_layer(module, parent_name="", pruning_layers=None, include_bn=False):
    if pruning_layers is None:
        pruning_layers = []

    for name, child in module.named_children():
        layer_full_name = f"{parent_name}.{name}" if parent_name else name
        
        if isinstance(child, nn.Conv2d):
            pruning_layers.append(layer_full_name)

        if isinstance(child, nn.Linear):
            pruning_layers.append(layer_full_name)
        
        if include_bn and isinstance(child, nn.BatchNorm2d):
            pruning_layers.append(layer_full_name)

        get_layer(child, layer_full_name, pruning_layers,include_bn)
    return pruning_layers

def get_module_by_name(model, path):
    parts = path.split('.')
    current_module = model

    for part in parts:
        current_module = getattr(current_module, part)

    return current_module

def get_del_indexes(model, pruning_layers, filter_num, args):
    all_filters = []
    filter_size = {}
    for layer_name in pruning_layers:
        module = get_module_by_name(model, layer_name)
        filters = module.weight.data.clone().detach()
        filter_size[layer_name] = filters.size(0) 
            
        sum_of_filters = torch.mean(torch.abs(filters.view(filters.size(0), -1)), dim=1)
        for i in range(sum_of_filters.size(0)):
            all_filters.append((sum_of_filters[i].item(), i, layer_name))

    all_filters.sort()

    threshold_index = int(len(all_filters) * args.per_iter_pruning_ratio)
    threshold_value = all_filters[threshold_index][0]

    result = {}
    for val, idx, name in all_filters:
        if name not in result:
            result[name] = []
        if val < threshold_value:
            result[name].append(idx)

    exclude = []
    for k in result.keys():
        if (filter_size[k] - len(result[k])) <= args.min_ratio * filter_num[k]:
            res = int(filter_size[k] - args.min_ratio * filter_num[k] )
            if res >= 0:
                result[k] = result[k][:res]
                exclude.append(k)
 
    return result, exclude


def reset_in_channel_index():
    global in_channel_index, save
    in_channel_index = []
    save = {}

in_channel_index, save = [], {}
def global_prune(module, result, parent_name="", residuals=[]):
    global in_channel_index, save
    for name, child in module.named_children():
        layer_full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(child, nn.Conv2d):
            out_channel_index = result.get(layer_full_name, [])

            if "downsample" in layer_full_name:
                idx = residuals.index(layer_full_name) - 2 
                if idx < 0:
                    in_channel_index = save["conv1"]
                else:
                    in_channel_index = save[residuals[idx]]

            new_conv = get_new_conv(child, in_channel_index, out_channel_index)
            setattr(module, name, new_conv)
            
            save[layer_full_name] = out_channel_index  
            in_channel_index = out_channel_index

        elif isinstance(child, nn.BatchNorm2d):
            new_ = get_new_norm(child, in_channel_index) 
            setattr(module, name, new_)

        elif isinstance(child, nn.Linear):
            
            new_linear = get_new_linear(child, in_channel_index)             
            setattr(module, name, new_linear)
            in_channel_index = []
        
        else:
            global_prune(child, result, layer_full_name, residuals)

    return module

@torch.no_grad()
def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()

    select_index = torch.tensor(list(set(range(tensor.size(dim))) - set(index)), dtype=torch.long)
    new_tensor = torch.index_select(tensor, dim, select_index)

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor



def get_new_conv(conv, in_channel_index, out_channel_index,):
    
    new_conv = torch.nn.Conv2d(
                                in_channels=int(conv.in_channels - len(in_channel_index)),
                                out_channels=int(conv.out_channels - len(out_channel_index)),
                                kernel_size=conv.kernel_size,
                                stride=conv.stride, 
                                padding=conv.padding, 
                                dilation=conv.dilation, 
                                groups=conv.groups, 
                                bias=conv.bias is not None, 
                                padding_mode=conv.padding_mode
                                )

    tem_weight = index_remove(conv.weight.data, 1, in_channel_index)
    new_conv.weight.data = index_remove(tem_weight, 0, out_channel_index)
    assert new_conv.weight.shape == (new_conv.out_channels, new_conv.in_channels, *new_conv.kernel_size)
    

    if conv.bias is not None:
        new_conv.bias.data = index_remove(conv.bias.data, 0, out_channel_index)

    return new_conv

def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)
        
    return new_norm

def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(
        in_features=int(linear.in_features - len(channel_index)),
        out_features=linear.out_features,
        bias=linear.bias is not None
    )
    new_weight = index_remove(linear.weight.data, 1, channel_index)
    new_linear.weight.data = new_weight
    assert new_linear.weight.shape == (new_linear.out_features, new_linear.in_features)

    if linear.bias is not None:
        new_linear.bias = linear.bias

    return new_linear


