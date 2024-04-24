# Basic Pruning for CNN with Pytorch
This repository offers a PyTorch reimplementation of a commonly used and practical method for pruning deep neural networks: magnitude pruning with granularity at the channel level. Our implementation is inspired by [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710), although it is not an exact replication.

## Features
- **Granularity**: Implements filter-wise pruning.
- **Importance Criterion**: Utilizes the mean of the L1 norm across filters.
- **Adaptive Pruning Ratios**: Chooses pruning ratios per layer adaptively.
- **Pruning Schedule**: iterative pruning.
- **Scope**: Focuses on global pruning 

## Goals
This repository is designed to help users understand and implement pruning within the PyTorch framework. It provides easy-to-understand code, demonstrations of model size reduction and speed increase.

## Requirements
- **PyTorch**: 2.2.1
- **timm**: 0.9.12

____________________________________________________________________________________________
## Run
Firstly, Download pretrained VGG16 frome here [Download](https://drive.google.com/file/d/1XFD5oe5QH_09lE4C-7yMIJ8O6OflPKOX/view?usp=sharing) and Run below code



For Sigle GPU
<pre>
python main.py --dataset "CIFAR10 CIFAR... " --data_path "path_to_data" --pretrained "path_to_pretrained_model" --device cuda --model vgg16 
</pre>

  
For Multi GPU
<pre>
torchrun --nnodes="number_of_nodes" --nproc_per_node="number_of_processes_per_node" main.py --dataset "CIFAR10 CIFAR... " --data_path "path_to_data" --pretrained "path_to_pretrained_model" --device cuda --model vgg16 --distributed 
</pre>


Modle trained with CIFAR10.
| Model Name           | Accuracy (%) | Size (MB) | Latency (ms) | Checkpoint |
|----------------------|--------------|-----------|--------------|------------|
| pytorch_vgg16  | 73.66        | 44.98     | 16.3784      |[Download](https://drive.google.com/file/d/1XFD5oe5QH_09lE4C-7yMIJ8O6OflPKOX/view?usp=sharing) |
| pytorch_vgg16_pruned  | ??      | ??     | ??     |[Download](https://drive.google.com/file/d/1B_cR5QlXdnpzGfaQcAGtFjV0d3kLctcJ/view?usp=sharing) |

