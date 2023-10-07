## Code Intro
The code in this directory implements two task extensions of V3Det, 
1. Open-Vocabulary Detection (OVD). 
2. Object Detection Pretraining. 

Here we adopt [CenterNet2](https://github.com/facebookresearch/Detic) as an example detector and replace its classifier with CLIP classifier to empower it with open-vocabulary capacity.

## Open-Vocabulary Detection
### OVD Base/Novel Split Preparation
Step 1. Download V3Det data place them in the following way:
```
datasets/
    metadata/
    V3det/
```

Step 2. Prepare the open-vocabulary V3Det training set using
````bash
python tools/v3det_ovd_utils/split_base_novel.py datasets/V3Det/annotations/v3det_2023_v1_train.json
````
this will generate `datasets/V3Det/annotations/v3det_2023_v1_train_ovd_base.json`

Step 3. Get category info json and clip classifier npy using
```bash
python tools/v3det_ovd_utils/get_cat_info.py --ann datasets/V3Det/annotations/v3det_2023_v1_train.json
python tools/v3det_ovd_utils/dump_clip_features.py --ann datasets/V3Det/annotations/v3det_2023_v1_train.json
```
this will generate category info json: `datasets/metadata/v3det_2023_v1_train_cat_info.json` and clip classifier npy `datasets/metadata/v3det_2023_v1_train_clip_a+cname.npy`

### Training and Evaluation
Training on single node with 8 GPUs:
````bash
python -u tools/train_detic.py --config-file projects/Detic/configs/ovd/BoxSup-C2_V3Det-OVD-Base_CLIP_R5021k_640b64_4x.yaml --num-gpus 8
````

evaluation:
````bash
python -u tools/train_detic.py --config-file projects/Detic/configs/ovd/BoxSup-C2_V3Det-OVD-Base_CLIP_R5021k_640b64_4x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS [model_path]
````

## Results 

    | Model                  | Seen AP50      | Unseen AP50    |
    |------------------------|----------------|----------------|
    | CenterNet2 CLIP        | 33.98          | 3.55           |


## Pretraining

### Dataset Preparation
Step 1. Download Objects365 and LVIS data place them in the following way:
```
datasets/
    metadata/
    V3det/
    Objects365/
    LVIS/
```

Step 2. Get category info json and clip classifier npy using
```bash
python tools/v3det_ovd_utils/get_cat_info.py --ann datasets/Objects365/objects365_train.json
python tools/v3det_ovd_utils/dump_clip_features.py --ann datasets/Objects365/objects365_train.json
python tools/v3det_ovd_utils/get_cat_info.py --ann datasets/lvis/annotations/lvis_v1_train.json
python tools/v3det_ovd_utils/dump_clip_features.py --ann datasets/lvis/annotations/lvis_v1_train.json
```
this will generate category info json and clip classifier npy

Step 3. Pretrain and Finetuning
```bash
# Pretrain on V3Det and finetune on LVIS
python -u tools/train_detic.py --config-file projects/Detic/configs/pretrain/BoxSup-C2_V3Det_CLIP_R5021k_640b64_4x.py --num-gpus 8
python -u tools/train_detic.py --config-file projects/Detic/configs/pretrain/BoxSup-C2_V3Det_LVIS_CLIP_R5021k_640b64_4x.py --num-gpus 8 MODEL.WEIGHTS [checkpoint_of_above]
```

```bash
# Pretrain on Objects365 and finetune on LVIS
python -u tools/train_detic.py --config-file projects/Detic/configs/pretrain/BoxSup-C2_Obj365_CLIP_R5021k_640b64_4x.py --num-gpus 8
python -u tools/train_detic.py --config-file projects/Detic/configs/pretrain/BoxSup-C2_Obj365_LVIS_CLIP_R5021k_640b64_4x.py --num-gpus 8 MODEL.WEIGHTS [checkpoint_of_above]
```

```bash
# Pretrain on Objects365, V3Det and finetune on LVIS
python -u tools/train_detic.py --config-file projects/Detic/configs/pretrain/BoxSup-C2_Obj365_V3Det_CLIP_R5021k_640b64_4x.py --num-gpus 8
python -u tools/train_detic.py --config-file projects/Detic/configs/pretrain/BoxSup-C2_Obj365_V3Det_LVIS_CLIP_R5021k_640b64_4x.py --num-gpus 8 MODEL.WEIGHTS [checkpoint_of_above]
```

## Results 
V3Det and Objects365 exhibit comparable performance and can complement each other for pretraining. Notable, Objects has 608606 training images, almost five times that of V3Det, which has 132437.

    | Pretrain Datasets      | LVIS Box mAP   |
    |------------------------|----------------|
    | None                   | 35.2           |
    | V3Det                  | 36.3           |
    | Objects365             | 36.8           |
    | V3Det + Objects365     | 37.7           |


## Citation

```latex
@article{wang2023v3det,
  title={V3det: Vast vocabulary visual detection dataset},
  author={Wang, Jiaqi and Zhang, Pan and Chu, Tao and Cao, Yuhang and Zhou, Yujie and Wu, Tong and Wang, Bin and He, Conghui and Lin, Dahua},
  journal={arXiv preprint arXiv:2304.03752},
  year={2023}
}
```