import copy

import torch

from prune import get_module_by_name

def get_residual(pruning_layers):

    residuals = []
    stages = { "layer1": ["conv1"], "layer2": [], "layer3": [], "layer4": []}
    layer1_downsample = False
    for i ,layer in enumerate(pruning_layers):
            
        if "layer1.0.downsample" in layer:
            layer1_downsample = True

        if "downsample" in layer:
            residuals.append(layer)
            continue

        try:
            if pruning_layers[i+1].split(".")[1] != pruning_layers[i].split(".")[1]:
                residuals.append(layer)
                continue

            if "downsample" in pruning_layers[i+1]:
                residuals.append(layer)
                continue
        except:
            pass

    residuals.append(pruning_layers[-1])

    #group by residual connected layers
    for stage in stages.keys():
        for i in range(len(residuals)):
            if stage in residuals[i]:
                stages[stage].append(residuals[i])

    
    if layer1_downsample is True:
        stages["layer1"].remove("conv1")

    return stages


def res_get_del_indexes(model, pruning_layers, filter_num, stages, args):
    all_filters = []
    filter_size = {}

    residuals = []
    for stage in stages.keys():
        for l in stages[stage]:
            residuals.append(l)

    non_res = []
    for layer in pruning_layers:
        if layer not in residuals:
            non_res.append(layer)

    for layer_name in non_res:
        module = get_module_by_name(model, layer_name)
        filters = module.weight.data.clone().detach()
        filter_size[layer_name] = filters.size(0) 
            
        sum_of_filters = torch.mean(torch.abs(filters.view(filters.size(0), -1)), dim=1)
        for i in range(sum_of_filters.size(0)):
            all_filters.append((sum_of_filters[i].item(), i, layer_name))

    for stage in stages.keys():
        mean_of_filters = 0
        for layer_name in stages[stage]:
            module = get_module_by_name(model, layer_name) 
            filters = module.weight.data.clone().detach()
            filter_size[layer_name] = filters.size(0) 
            
            mean_of_filters += torch.mean(torch.abs(filters.view(filters.size(0), -1)), dim=1)
        if len(stages[stage]) == 0:
            continue
        mean_of_filters = mean_of_filters / len(stages[stage])
        
        for i in range(mean_of_filters.size(0)):
            all_filters.append((mean_of_filters[i].item(), i, stage))


    all_filters.sort()

    threshold_index = int(len(all_filters) *args.per_iter_pruning_ratio)
    threshold_value = all_filters[threshold_index][0]

    result = {}
    for val, idx, name in all_filters:
        if name not in result:
            result[name] = []
        if val < threshold_value:
            result[name].append(idx)

    exclude = []
    for k in result.keys():
        try:
            fil_nums = filter_num[k]
            fil_size = filter_size[k]
        except:
            fil_nums = filter_num[stages[k][0]]
            fil_size = filter_size[stages[k][0]]
        if (fil_size - len(result[k])) <= args.min_ratio* fil_nums:
            res = int(fil_size - args.min_ratio * fil_nums )
            if res >= 0:
                result[k] = result[k][:res]
                exclude.append(k)

    tem_exclude = copy.deepcopy(exclude)
    for exclude_layer in tem_exclude:
        if exclude_layer in stages.keys():
            for e in stages[exclude_layer]:
                exclude.append(e)
            exclude.remove(exclude_layer)
    del tem_exclude 
    
    tem_result = copy.deepcopy(result)
    for k in tem_result.keys():
        if k in stages.keys():
            for e in stages[k]:
                result[e] = result[k] 
            del result[k]
    del tem_result

    return result, exclude
