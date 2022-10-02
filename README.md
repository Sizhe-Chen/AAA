# Decription
* The code is the official implementation of NeurIPS paper [Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks](https://arxiv.org/abs/2205.12134)
* This repository supports data protection on CIFAR-10 and ImageNet
* The experiments are run in an NVIDIA A100 GPU, but could modify the batch size to run on small GPUs
* Install dependencies
```
conda env create -f pt.yaml
```


# Reproduction
* train the protecting DNN

```
python vanilla.py
```
```
python vanilla100.py
```
```
python vanillaimg.py
```

* crafting protective samples (CIFAR-10, SEP)

```
python ens.py --num_model=30 --eps=2 --target_batch=0
```

* crafting protective samples (CIFAR-10, SEP-FA)

```
python ens_feature.py --num_model=30 --eps=2 --target_batch=0
```

* crafting protective samples (CIFAR-10, SEP-FA-VR)

```
python ens_feature_svre.py --num_model=15 --eps=2 --target_batch=0
```

* crafting protective samples (CIFAR-100, SEP-FA-VR)

```
python ens_feature_svre100.py --num_model=15 --eps=2 --target_batch=0
```

* crafting protective samples (ImageNet subset, SEP-FA-VR)

```
python ens_feature_svreimg.py --num_model=15 --eps=2 --target_batch=0
```

* train the appropriator DNN
```
python vanilla.py --uledir=samples/XX --eps=2
```
```
python vanilla100.py --uledir=samples/XX --eps=2
```
```
python vanillaimg.py --uledir=samples/XX --eps=2
```

# Files
```
├── ens_feature.py
```
