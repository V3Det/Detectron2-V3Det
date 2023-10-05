# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import math

import mmengine
import numpy as np
from detectron2.utils.logger import create_small_table
from tabulate import tabulate

from .coco_evaluation import COCOEvaluator


class CustomV3DetEvaluator(COCOEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cats = mmengine.load(
            'datasets/metadata/v3det_2023_v1_train_cat_info.json')
        # cats = sorted(cats, key=lambda x: x['id'])
        categories_seen_inds = []
        categories_unseen_inds = []
        for i, c in enumerate(cats):
            if c['novel']:
                categories_unseen_inds.append(i)
            else:
                categories_seen_inds.append(i)
        self.categories_seen_inds = set(categories_seen_inds)
        self.categories_unseen_inds = set(categories_unseen_inds)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Additionally plot mAP for 'seen classes' and 'unseen classes'
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] *
                          100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info("Evaluation results for {}: \n".format(iou_type) +
                          create_small_table(results))
        if not np.isfinite(sum(results.values())):
            self._logger.info(
                "Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_seen = []
        results_per_category_unseen = []
        results_per_category50 = []
        results_per_category50_seen = []
        results_per_category50_unseen = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            precision50 = precisions[0, :, idx, 0, -1]
            precision50 = precision50[precision50 > -1]
            ap50 = np.mean(precision50) if precision50.size else float("nan")
            results_per_category50.append(
                ("{}".format(name), float(ap50 * 100)))
            if idx in self.categories_seen_inds and (not math.isnan(ap)):
                results_per_category_seen.append(float(ap * 100))
                results_per_category50_seen.append(float(ap50 * 100))
            if idx in self.categories_unseen_inds and (not math.isnan(ap50)):
                results_per_category_unseen.append(float(ap * 100))
                results_per_category50_unseen.append(float(ap50 * 100))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        # self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        N_COLS = min(6, len(results_per_category50) * 2)
        results_flatten = list(itertools.chain(*results_per_category50))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        # self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)
        self._logger.info("Seen {} AP50: {}".format(
            iou_type,
            sum(results_per_category50_seen) /
            len(results_per_category50_seen),
        ))
        self._logger.info("Seen {} AP: {}".format(
            iou_type,
            sum(results_per_category_seen) / len(results_per_category_seen),
        ))
        self._logger.info("Unseen {} AP50: {}".format(
            iou_type,
            sum(results_per_category50_unseen) /
            len(results_per_category50_unseen),
        ))
        self._logger.info("Unseen {} AP: {}".format(
            iou_type,
            sum(results_per_category_unseen) /
            len(results_per_category_unseen),
        ))

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        results["AP50-seen"] = sum(results_per_category50_seen) / len(
            results_per_category50_seen)
        results["AP50-unseen"] = sum(results_per_category50_unseen) / len(
            results_per_category50_unseen)
        return results
