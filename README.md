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

## Available Models
- VGG16 [Link to paper]

## Usage Instructions
### Multi-GPU Setup:
```bash
torchrun --nnodes=<number_of_nodes> --nproc_per_node=<number_of_processes_per_node> main.py --dataset CIFAR10 --data_path "<path_to_data>" --pretrained "<path_to_pretrained_model>" --device cuda --epochs <number_of_epochs> --pruning_ratio 0.95 --model vgg16 --lr 0.0001 --per_iter_pruning_ratio 0.05 --print_freq 500 --distributed




# Basic Pruning with Pytorch
This repository offers a PyTorch reimplementation of the pruning approach that most common and practical method which magnitude pruning with granularity with channels. you can find our code built behind theory with paper but not perfectly same.  
PRUNING FILTERS FOR EFFICIENT CONVNETS 


## Global Magnitude pruning for CNN.
(1) granularity: filter wise pruning
(2) importance criterion : mean of L1 norm in filter
(3) how to choose per layer pruning ratio: adaptively 
(4) pruning schedule: iterative pruning
(5) pruning scope : Global pruning

## Goal
This repository is dedicated to those who wish to comprehend the process of quantization within the PyTorch framework. It offers a straightforward and accessible implementation, as well as practical demonstrations of model size reduction and speed up with ONNX inference. 


## Requirements
- PyTorch: 2.2.1
- CUDA: 12.1
- timm: 0.9.12
- ONNX: 1.16.0
- ONNX Runtime: 1.17.1

## available models
-vgg 16 [paper]

____________________________________________________________________________________________
## Run with 
For multi GPU
<pre>
torchrun --nnodes="" --nproc_per_node="" main.py --data-set CIFAR10 --data_path "data_path" --pretrained "pretrained_model_path" --device cuda --epochs  --pruning_ratio 0.95 --model vgg16 --lr 0.0001 --per_iter_pruning_ratio 0.05 --print_freq 500 --distributed
</pre>
For single GPU

<pre>
torchrun --nnodes=1 --nproc_per_node=1 main.py --data-set CIFAR10 --data_path "data_path" --pretrained "pretrained_model_path" --device cuda --epochs  --pruning_ratio 0.95 --model vgg16 --lr 0.0001 --per_iter_pruning_ratio 0.05 --print_freq 500 
</pre>

Modle trained with CIFAR10 and calibrate with CIFAR10.
| Model Name           | Accuracy (%) | Size (MB) | Latency (ms) | Checkpoint |
|----------------------|--------------|-----------|--------------|------------|
| pytorch_resnet18_fp  | 73.66        | 44.98     | 16.3784      |[Download](https://drive.google.com/file/d/1DXdomOlWoPvT2DKW6_r9tq9v2rH8y_00/view?usp=sharing) |
| onnx_resnet18_quant  | 73.52        | 11.35     | 10.2182      |[Download](https://drive.google.com/file/d/1B_cR5QlXdnpzGfaQcAGtFjV0d3kLctcJ/view?usp=sharing) |

