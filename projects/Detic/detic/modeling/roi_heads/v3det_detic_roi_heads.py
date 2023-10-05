# Copyright (c) Facebook, Inc. and its affiliates.

import torch

from detectron2.layers import batched_nms
from detectron2.modeling.box_regression import Box2BoxTransform
# from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
# from detic.modeling.utils import perm_boxes
from .detic_roi_heads import DeticCascadeROIHeads
from .v3det_detic_fast_rcnn import V3DetDeticFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class V3DetDeticCascadeROIHeads(DeticCascadeROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], \
            cascade_bbox_reg_weights):
            box_predictors.append(
                V3DetDeticFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(
                        weights=bbox_reg_weights)))
        ret['box_predictors'] = box_predictors
        return ret

    def _forward_box(self,
                     features,
                     proposals,
                     targets=None,
                     ann_type='box',
                     classifier_info=(None, None, None)):
        """
        Add mult proposal scores at testing
        Add ann_type
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals
                ]

        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes,
                    image_sizes,
                    logits=[p.objectness_logits for p in proposals])
                if self.training and ann_type in ['box']:
                    proposals = self._match_and_label_boxes(
                        proposals, k, targets)
            predictions = self._run_stage(features,
                                          proposals,
                                          k,
                                          classifier_info=classifier_info)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append(
                (self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions,
                        proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    if ann_type != 'box':
                        stage_losses = {}
                        if ann_type in ['image', 'caption', 'captiontag']:
                            image_labels = [
                                x._pos_category_ids for x in targets
                            ]
                            weak_losses = predictor.image_label_losses(
                                predictions,
                                proposals,
                                image_labels,
                                classifier_info=classifier_info,
                                ann_type=ann_type)
                            stage_losses.update(weak_losses)
                    else:  # supervised
                        stage_losses = predictor.losses(
                            (predictions[0], predictions[1]),
                            proposals,
                            classifier_info=classifier_info)
                        if self.with_image_labels:
                            stage_losses['image_loss'] = \
                                predictions[0].new_zeros([1])[0]
                losses.update({k + "_stage{}".format(stage): v \
                               for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [
                h[0].predict_probs(h[1], h[2]) for h in head_outputs
            ]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                          for s, ps in zip(scores, proposal_scores)]
            if self.one_class_per_proposal:
                scores = [
                    s * (s == s[:, :-1].max(dim=1)[0][:, None]).float()
                    for s in scores
                ]
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes((predictions[0], predictions[1]),
                                            proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances


def fast_rcnn_inference(
    boxes,
    scores,
    image_shapes,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    result_per_image = [
        fast_rcnn_inference_single_image(boxes_per_image, scores_per_image,
                                         image_shape, score_thresh, nms_thresh,
                                         topk_per_image)
        for scores_per_image, boxes_per_image, image_shape in zip(
            scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(
        dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    num_topk = min(10000, scores.size(0))
    pre_topk = scores.topk(num_topk).indices
    boxes_topk = boxes[pre_topk]
    scores_topk = scores[pre_topk]
    filter_inds_topk = filter_inds[pre_topk]
    keep_topk = batched_nms(boxes_topk, scores_topk, filter_inds_topk[:, 1],
                            nms_thresh)
    keep = pre_topk[keep_topk]
    if keep.size(0) < topk_per_image:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    # keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    # keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    # if topk_per_image >= 0:
    #     keep = keep[:topk_per_image]
    # boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]
