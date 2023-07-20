from functools import partial

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

from .v3det_loader_lsj_1280 import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher", "mask_in_features", "mask_pooler", "mask_head"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    num_classes=13029,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            num_classes="${...num_classes}",
            test_score_thresh=0.02,
            test_topk_before_nms=100000,  #used for vast classes
            test_topk_per_image=300,
            cls_agnostic_bbox_reg=True,
            use_sigmoid_ce=True,
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

model.backbone.net.beit_like_qkv_bias = True
model.backbone.net.beit_like_gamma = False
model.backbone.net.freeze_patch_embed = True
model.backbone.square_pad = 1280
model.backbone.net.img_size = 1280
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1408
model.backbone.net.depth = 40
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 6144 / 1408
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.6  # 0.5 --> 0.6
# global attention for every 4 blocks
model.backbone.net.window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)) + list(range(28, 31)) + list(range(32, 35)) + list(range(36, 39))
)

optimizer.lr = 2.5e-5
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.9, num_layers=40)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None


# Schedule
# 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
#train.max_iter = 70001
train.max_iter = 70001000000
train.eval_period = 70002

lr_multiplier.scheduler.milestones = [65000, 70000]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4
