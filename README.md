# Decription
* The code is the official implementation of NeurIPS paper [Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks](https://neurips.cc/virtual/2022/poster/54907)
* This repository supports data protection on CIFAR-10 and ImageNet
* The experiments are run in an NVIDIA A100 GPU, but could modify the batch size to run on small GPUs
* Install dependencies
```
conda env create -f pt.yaml
```
* Prepare [ImageNet validation set (2012)](http://www.image-net.org), place in folder 'data/ILSVRC2012_img_val'


# Numerical results of AAA (Table 2)
* CIFAR-10 (WideResNet28)
```
python square.py
```
```
python square.py --model=Dai2021Parameterizing
```
```
python square.py --defense=inRND
```
```
python square.py --defense=AAALinear
```

* ImageNet (WideResNet50)
```
python square.py --dataset=imagenet --model=wide_resnet50_2 --eps=4
```
```
python square.py --dataset=imagenet --model=Salman2020Do_50_2 --eps=4
```
```
python square.py --dataset=imagenet --model=wide_resnet50_2 --defense=inRND --eps=4
```
```
python square.py --dataset=imagenet --model=wide_resnet50_2 --defense=AAALinear --eps=4
```

* ImageNet (ResNeXt101)
```
python square.py --dataset=imagenet --model=resnext101_32x8d --eps=4
```
```
python square.py --dataset=imagenet --model=ResNeXt101_DenoiseAll --eps=4
```
```
python square.py --dataset=imagenet --model=resnext101_32x8d --defense=inRND --eps=4
```
```
python square.py --dataset=imagenet --model=resnext101_32x8d --defense=AAALinear --eps=4
```

# Generalization of AAA (Table 4)
* vanilla training
```
python square.py --targeted --defense=AAALinear
```
```
python square.py --l2 --eps=0.5
```
* adversarial training
```
python square.py --targeted --defense=AAALinear --model=Dai2021Parameterizing
```
```
python square.py --l2 --eps=0.5 --model=Dai2021Parameterizing
```

# Adaptive attacks of AAA (Table 6)
* bi-Square
```
python square.py --loss=bi --defense=AAALinear
```
```
python square.py --loss=bi --defense=AAASine
```
* up-Square
```
python square.py --loss=up --defense=AAALinear
```
```
python square.py --loss=up --defense=AAASine
```

# Others
* QueryNet attack (Table 3)
```
python square.py --num_s=3 --gpu=0,1,2,3 --defense=AAALinear
```
* Attack by CE-loss (Table 8)
```
python square.py --loss=ce --defense=AAALinear
```

# Files
```
├── attacker.py
├── data
│   └── val.txt # for ImageNet attacks
├── dent.py
├── dfdmodels # for ResNeXt101_DenoiseAll
│   ├── adv_model.py
│   ├── inception_resnet_v2.py
│   ├── __init__.py
│   ├── nets.py
│   ├── resnet_model.py
│   └── third_party
│       ├── imagenet_utils.py
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── imagenet_utils.cpython-36.pyc
│       │   ├── imagenet_utils.cpython-37.pyc
│       │   ├── __init__.cpython-36.pyc
│       │   └── __init__.cpython-37.pyc
│       ├── README.md
│       ├── serve-data.py
│       └── utils.py
├── PCDARTS # for querynet attack
│   ├── architect.py
│   ├── genotypes.py
│   ├── __init__.py
│   ├── model.py
│   ├── model_search_imagenet.py
│   ├── model_search.py
│   ├── model_search_random.py
│   ├── operations.py
│   ├── README.md
│   ├── test.py
│   ├── train_imagenet.py
│   ├── train.py
│   ├── train_search_imagenet.py
│   ├── train_search.py
│   ├── utils.py
│   ├── V100_python1.0
│   │   ├── train.py
│   │   └── train_search.py
│   └── visualize.py
├── pt.yaml
├── square.py
├── utils.py
└── victim.py
```
