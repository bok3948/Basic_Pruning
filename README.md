# Basic Pruning with Pytorch
This repository offers a PyTorch reimplementation of a commonly used and practical method for pruning deep neural networks: magnitude pruning with granularity at the channel level. Our implementation, while inspired by foundational theory from the literature, is uniquely adapted for practical applications.

Reference: [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) by Pavlo Molchanov et al.

## Features
- **Granularity**: Implements filter-wise pruning.
- **Importance Criterion**: Utilizes the mean of the L1 norm across filters.
- **Adaptive Pruning Ratios**: Chooses pruning ratios per layer adaptively.
- **Pruning Schedule**: Supports iterative pruning.
- **Scope**: Focuses on global pruning across the network.

## Goals
This repository is designed to help users understand and implement pruning within the PyTorch framework. It provides easy-to-understand code, demonstrations of model size reduction and speed increase.

## Requirements
- **PyTorch**: 2.2.1
- **CUDA**: 12.1
- **timm**: 0.9.12
- **ONNX**: 1.16.0
- **ONNX Runtime**: 1.17.1

## Usage Instructions
<pre>
torchrun --nnodes=<number_of_nodes> --nproc_per_node=<number_of_processes_per_node> main.py --dataset "CIFAR10 CIFAR... " --data_path "<path_to_data>" --pretrained "path_to_pretrained_model" --device cuda --model vgg16 --distributed 
</pre>


Modle trained with CIFAR10 and calibrate with CIFAR10.
| Model Name           | Accuracy (%) | Size (MB) | Latency (ms) | Checkpoint |
|----------------------|--------------|-----------|--------------|------------|
| pytorch_vgg16  | 73.66        | 44.98     | 16.3784      |[Download](https://drive.google.com/file/d/1DXdomOlWoPvT2DKW6_r9tq9v2rH8y_00/view?usp=sharing) |
| pytorch_vgg16_pruned  | 73.52        | 11.35     | 10.2182      |[Download](https://drive.google.com/file/d/1B_cR5QlXdnpzGfaQcAGtFjV0d3kLctcJ/view?usp=sharing) |

