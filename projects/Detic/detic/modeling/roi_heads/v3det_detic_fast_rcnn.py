# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch.nn import functional as F

from detic.data.detection_utils import get_fed_loss_cls_weights
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from ..utils import get_fed_loss_inds

__all__ = ["V3DetDeticFastRCNNOutputLayers"]


class V3DetDeticFastRCNNOutputLayers(DeticFastRCNNOutputLayers):
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["get_fed_loss_cls_weights"] = lambda: get_fed_loss_cls_weights(
            dataset_names=cfg.DATASETS.TRAIN,
            freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER
        )
        return ret

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros(
                [1])[0]  # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1  # B x (C + 1)
        target = target[:, :C]  # B x C

        weight = 1

        if self.use_fed_loss and (self.freq_weight is not None):  # fedloss
            appeared = get_fed_loss_inds(gt_classes,
                                         num_sample_cats=self.fed_loss_num_cat,
                                         C=C,
                                         weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1  # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()
        if self.ignore_zero_cats and (self.freq_weight is not None):
            w = (self.freq_weight.view(-1) > 1e-4).float()
            weight = weight * w.view(1, C).expand(B, C)
            # import pdb; pdb.set_trace()

        ignore_weight = 1
        if self.num_classes == 13082:  # include coarse classes
            ignore_cls = list([self.num_classes - i - 1
                               for i in range(53)])[::-1]
            ignore_cls = gt_classes.new_tensor(ignore_cls)
            ignore_rois_inds = (
                gt_classes[:, None] == ignore_cls[None, :]).sum(-1).bool()
            ignore_weight = torch.ones_like(target).detach().float()
            ignore_weight[ignore_rois_inds] = 0

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')  # B x C
        loss = torch.sum(cls_loss * weight * ignore_weight) / B
        return loss
