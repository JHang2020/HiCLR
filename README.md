# HiCLR

This is an official PyTorch implementation of [**"Hierarchical Consistent Contrastive Learning for Skeleton-Based Action Recognition with Growing Augmentations"**](https://arxiv.org/abs/2211.13466) in *AAAI 2023*. 

## Requirements
  ![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)    ![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.4-blue.svg)


## Data Preparation
- Download the raw data of [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) and [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).
- For NTU RGB+D dataset, preprocess data with `tools/ntu_gendata.py`. For PKU-MMD dataset, preprocess data with `tools/pku_part1_gendata.py`.
- Then downsample the data to 50 frames with `feeder/preprocess_ntu.py` and `feeder/preprocess_pku.py`.

## Train the Model
See the run_cs.sh for the detail instructions.

You can change the settings in the corresponding `.yaml` file. 

```bash
# train on NTU RGB+D xsub joint stream
$ python main.py pretrain_hiclr --config config/ntu60/pretext/pretext_hiclr_xsub_joint.yaml
```
## Results and Pre-trained Models
For three-streams results, we use the code in ensemble_xxx.py to obtain the fusion results.
The performance of the released code is better than that reported in the paper.
You can find the pre-trained model weights here.

|     Model     | NTU 60 xsub (%) |
| :-----------: | :-------------: |
| HiCLR-joint  |      77.30      |
| HiCLR-motion |      70.29      |
|  HiCLR-bone  |            |
|   3s-HiCLR   |    **80.**    |


## Acknowledgement
We sincerely thank the authors for releasing the code of their valuable works. Our code is built based on the following repos.
- The code of our framework is heavily based on [AimCLR](https://github.com/Levigty/AimCLR).
- The code of encoder is based on [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).

## Licence

This project is licensed under the terms of the MIT license.
