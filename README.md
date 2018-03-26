# PyTorch-Template

This project borrows ideas from [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template), 
aimming to provide a easy and extensible template for pytorch projects.


## Structure
```
├── datasets 
        (wrapper for datasets)
├── mains
        (main.py for various datasets and tasks)
├── models
        (network descriptions)
├── options
        (template parser for command line inputs)
├── trainers
        (define how model's forward / backward and logs)
├── utils
        (toolbox for drawing and scheduling)
├── scripts
        (scripts for replicating experiement results)
└── work
        (default folder to store logs/models)
```

## Usage examples
*  Train ResNet-20 (pre-activation) on CIFAR100
    ```bash
    python main/cifar.py \
           ~/torch_data/cifar \
           --arch resnet20 \
           --dataset cifar100 
    ```
    
* Train ResNet-18 on ImageNet
    ```bash
    python main/imagenet.py \
           ~/torch_data/imagenet \
           --arch resnet18 
    ```