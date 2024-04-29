# Basic Pruning for CNN with Pytorch
This repository offers a PyTorch reimplementation of a commonly used and practical method for pruning deep neural networks: magnitude pruning with granularity at the channel level. Our implementation is inspired by [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710), it is not a direct replication.

![Example Image](/images/acc_prune_trade_off.png "Example Image Titl")

## Features
- **Granularity**: Implements filter-wise pruning.
- **Importance Criterion**: Utilizes the mean of the L1 norm across filters.
- **Adaptive Pruning Ratios**: Chooses pruning ratios per layer adaptively.
- **Pruning Schedule**: iterative pruning.
- **Scope**: Focuses on global pruning 

## Requirements
- **PyTorch**: 2.2.1
- **timm**: 0.9.12

____________________________________________________________________________________________
## Run

For Sigle GPU
<pre>
python main.py --dataset CIFAR10 --data_path "path_to_data" --pretrained "path_to_pretrained_model" --device cuda --model resnet18 --pruning_ratio 0.7 --per_iter_pruning_ratio 0.05 --min_ratio 0.01
</pre>

  
For Multi GPU
<pre>
torchrun --nnodes="number_of_nodes" --nproc_per_node="number_of_processes_per_node" main.py --dataset "CIFAR10 CIFAR... " --data_path "path_to_data" --pretrained "path_to_pretrained_model" --device cuda --model vgg16 --distributed 
</pre>


Modle trained with CIFAR10. our Results Flop calculated with input size 1x3x32x32.
| Model Name                   | Accuracy (%) | #Parameters (M) | MFLOPs | Size (MB) | Latency (ms) | Checkpoint | Args |
|------------------------------|--------------|-----------------|--------|-----------|--------------|------------|------|
| pytorch_resnet18             | 86.42        | 11.18           | 37.12  | 44.8      | 3.1154       | [Download](https://drive.google.com/file/d/1iR6WdiGQ1ceWspa_jppUvklgK39k13NH/view?usp=sharing) |  None    |
| pytorch_pruned_resnet18      | 85.98        | 3.00            | 28.94  | 12.07     | 1.9616       | [Download](https://drive.google.com/file/d/1Gz0sbNiMQhzRJ7GmypVDSJ7sCvsg8-h0/view?usp=sharing) | [Download](https://drive.google.com/file/d/1my4jlBBzItb1noBnwAehYnQUotnlCypo/view?usp=sharing)  |
| pytorch_resnet34             | 86.66        | 21.28           | 74.92  | 85.29     | 4.505        | [Download](https://drive.google.com/file/d/1_eipZl72oBA0vBYIVwNoX1IZj5HHWk_U/view?usp=sharing) |  None  |
| pytorch_pruned_resnet34      | 86.53        | 5.65            | 50.99  | 22.73     | 2.6352       | [Download](https://drive.google.com/file/d/1EDMLssNLoS3Nz5NEqouHER04TiiAz9Xo/view?usp=sharing) |   [Download](https://drive.google.com/file/d/1mmaQ7hxKyD9bz_44sp4MV4xgaa9FvH02/view?usp=sharing)    |
| pytorch_resnet50             | 87.27        | 23.52           | 83.89  | 94.41     | 5.4016       | [Download](https://drive.google.com/file/d/12UjAI5H0haUCt-JBoQO77ADMfTbdIfGh/view?usp=sharing) |  None   |
| pytorch_pruned_resnet50      | 85.70        | 6.31            | 52.88  | 25.48     | 3.9642       | [Download](https://drive.google.com/file/d/1r5TXTT_3_u8wF-e13g2PmXcFpSkniJtN/view?usp=sharing) |   [Download](https://drive.google.com/file/d/1D71iP-Euhgszu1C5eBkfd0Rr14ShpJDY/view?usp=sharing)    |



