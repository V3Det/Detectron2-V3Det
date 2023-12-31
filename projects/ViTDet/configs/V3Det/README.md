<p>
<div align="center">

# <img src="v3det_icon.jpg" height="25"> V3Det: Vast Vocabulary Visual Detection Dataset

<div>
    <a href='https://myownskyw7.github.io/' target='_blank'>Jiaqi Wang</a>*,
    <a href='https://panzhang0212.github.io/' target='_blank'>Pan Zhang</a>*,
    Tao Chu*,
    Yuhang Cao*, </br>
    Yujie Zhou,
    <a href='https://wutong16.github.io/' target='_blank'>Tong Wu</a>,
    Bin Wang,
    Conghui He,
    <a href='http://dahua.site/' target='_blank'>Dahua Lin</a></br>
    (* equal contribution)</br>
    <strong>Accepted to ICCV 2023 (Oral)</strong>
</div>
</p>
<p>
<div>
    <strong>
        <a href='https://arxiv.org/abs/2304.03752' target='_blank'>Paper</a>,
        <a href='https://v3det.openxlab.org.cn/' target='_blank'>Dataset</a></br>
    </strong>
</div>
</div>
</p>

<div align=center>
    <img width=960 src="https://github.com/open-mmlab/mmdetection/assets/17425982/9c216387-02be-46e6-b0f2-b856f80f6d84"/>
</div>

<!-- [ALGORITHM] -->

## Abstract

Recent advances in detecting arbitrary objects in the real world are trained and evaluated on object detection datasets with a relatively restricted vocabulary. To facilitate the development of more general visual object detection, we propose V3Det, a vast vocabulary visual detection dataset with precisely annotated bounding boxes on massive images. V3Det has several appealing properties: 1) Vast Vocabulary: It contains bounding boxes of objects from 13,204 categories on real-world images, which is 10 times larger than the existing large vocabulary object detection dataset, e.g., LVIS. 2) Hierarchical Category Organization: The vast vocabulary of V3Det is organized by a hierarchical category tree which annotates the inclusion relationship among categories, encouraging the exploration of category relationships in vast and open vocabulary object detection. 3) Rich Annotations: V3Det comprises precisely annotated objects in 243k images and professional descriptions of each category written by human experts and a powerful chatbot. By offering a vast exploration space, V3Det enables extensive benchmarks on both vast and open vocabulary object detection, leading to new observations, practices, and insights for future research. It has the potential to serve as a cornerstone dataset for developing more general visual perception systems. V3Det is available at https://v3det.openxlab.org.cn/.


## Prepare Dataset

Please download and prepare V3Det Dataset at [V3Det Homepage](https://v3det.openxlab.org.cn/) and [V3Det Github](https://github.com/V3Det/V3Det).

The data includes a training set, a validation set, comprising 13,204 categories. The training set consists of 183,354 images, while the validation set has 29,821 images. The data organization is:

```
data/
    V3Det/
        images/
            <category_node>/
                |────<image_name>.png
                ...
            ...
        annotations/
            |────v3det_2023_v1_category_tree.json       # Category tree
            |────category_name_13204_v3det_2023_v1.txt  # Category name
            |────v3det_2023_v1_train.json               # Train set
            |────v3det_2023_v1_val.json                 # Validation set
```

## Training
Please follow the [EVA](https://github.com/baaivision/EVA/tree/master/EVA-01/det) to build the Detectron2.

### EVA
Slurm training on 4 nodes (32 A100):
````bash
srun -p cluster --cpus-per-task=112 --gres=gpu:8 --ntasks=4 --ntasks-per-node=1 --job-name=eva1280 multi-4node_run.sh --config-file projects/ViTDet/configs/V3Det/cascade_mask_rcnn_vitdet_eva_1280.py "train.init_checkpoint=eva_o365.pth" "train.output_dir=output2/v3det1280"
````

evaluation on 8 A100:
````bash
python tools/lazyconfig_train_net.py --num-gpus 8 --eval-only --config-file projects/ViTDet/configs/V3Det/cascade_mask_rcnn_vitdet_eva_1536.py "dataloader.evaluator.output_dir=output2/v3det1280_eval" "train.init_checkpoint=output2/v3det1280/model_final.pth"
````

### CenterNet2
Training on a single node with 8 gpus:
```bash
python -u tools/train_detic.py --config-file projects/Detic/configs/BoxSup-C2_V3Det_CLIP_R5021k_640b64_4x.yaml --num-gpus 8
```

Evaluation on a single node with 8 gpus:
```bash
python -u tools/train_detic.py --config-file projects/Detic/configs/BoxSup-C2_V3Det_CLIP_R5021k_640b64_4x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS [model_path]
```

## Results and Models

| Backbone  |      Model      | Lr schd | box AP |                                       Config                                        |                                   Download                                   |
|:---------:| :-------------: |:-------:|:------:|:-----------------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
|    EVA    |  Cascade R-CNN   |   2x    |  49.4  |   [config](./projects/ViTDet/configs/V3Det/cascade_mask_rcnn_vitdet_eva_1536.py)    | [model](https://download.openxlab.org.cn/models/V3Det/V3Det/weight/eva_1280) |
| CenterNet |  Cascade R-CNN   |   4x    |  30.1  | [config](./projects/Detic/configs/BoxSup-C2_V3Det_CLIP_R5021k_640b64_4x.yaml) |                                  [model](-)                                  |

## Open-Vocabulary Detection
We also provide code implementation of Open-Vocabulary Object Detection on V3Det. Please refer to [projects/Detic](projects/Detic) for details.

## Citation

```latex
@inproceedings{wang2023v3det,
      title = {V3Det: Vast Vocabulary Visual Detection Dataset}, 
      author = {Wang, Jiaqi and Zhang, Pan and Chu, Tao and Cao, Yuhang and Zhou, Yujie and Wu, Tong and Wang, Bin and He, Conghui and Lin, Dahua},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      month = {October},
      year = {2023}
}
```
