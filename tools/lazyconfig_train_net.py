#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import os
import logging

#os.environ['DETECTRON2_DATASETS'] = '../../../Data/LVIS'
os.environ['DETECTRON2_DATASETS'] = '../../../Data/V3Det'
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    train_loader.dataset.dataset.dataset[0]

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter,
                                use_wandb=args.wandb),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        do_test(cfg, model)
        #print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


#python tools/lazyconfig_train_net.py --num-gpus 8 --eval-only --config-file projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva_1536.py "dataloader.evaluator.output_dir=output2/pretrain/lvis_eval" "train.init_checkpoint=output2/pretrain/eva_lvis.pth" "dataloader.evaluator.max_dets_per_image=300" "model.roi_heads.maskness_thresh=0.5"
#python tools/lazyconfig_train_net.py --num-gpus 8 --eval-only --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva_1536.py "dataloader.evaluator.output_dir=output2/pretrain/coco_eval" "train.init_checkpoint=output2/pretrain/eva_coco_det.pth" "model.roi_heads.use_soft_nms=True" 'model.roi_heads.method="linear"' "model.roi_heads.iou_threshold=0.6" "model.roi_heads.override_score_thresh=0.0"

#lvis eva
#srun -p labshare2 --cpus-per-task=192 --gres=gpu:8 --ntasks=2 --ntasks-per-node=1 --job-name=eva multi-node_run.sh --config-file projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva.py "train.init_checkpoint=eva_o365.pth" "train.output_dir=output"
#srun -p mllm --cpus-per-task=112 --gres=gpu:8 --ntasks=4 --ntasks-per-node=1 --job-name=lvis multi-4node_run.sh --config-file projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva.py "train.init_checkpoint=eva_o365.pth" "train.output_dir=output2/lvis1280"
#lvis eva with v3det pretrain
#srun -p mllm --cpus-per-task=112 --gres=gpu:8 --ntasks=4 --ntasks-per-node=1 --job-name=lvis multi-4node_run.sh --config-file projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva.py "train.init_checkpoint=output2/v3det1280/model_final.pth" "train.output_dir=output2/v3det1280_lvis"


#v3det eva
#python tools/lazyconfig_train_net.py --num-gpus 1 --config-file projects/ViTDet/configs/V3Det/cascade_mask_rcnn_vitdet_eva.py "train.init_checkpoint=eva_o365.pth" "train.output_dir=output2/v3det640"
#srun -p labshare2 --cpus-per-task=192 --gres=gpu:8 --ntasks=2 --ntasks-per-node=1 --job-name=eva_test multi-node_run.sh --eval-only --config-file projects/ViTDet/configs/V3Det/cascade_mask_rcnn_vitdet_eva_1536.py "train.init_checkpoint=output2/v3det640/model_final.pth"

#v3det eva 1280
#srun -p labshare2 --cpus-per-task=112 --gres=gpu:8 --ntasks=4 --ntasks-per-node=1 --job-name=eva1280 multi-4node_run.sh --config-file projects/ViTDet/configs/V3Det/cascade_mask_rcnn_vitdet_eva_1280.py "train.init_checkpoint=eva_o365.pth" "train.output_dir=output2/v3det1280"
#python tools/lazyconfig_train_net.py --num-gpus 8 --eval-only --config-file projects/ViTDet/configs/V3Det/cascade_mask_rcnn_vitdet_eva_1536.py "dataloader.evaluator.output_dir=output2/v3det1280_eval" "train.init_checkpoint=output2/v3det1280/model_final.pth"

#python tools/lazyconfig_train_net.py --num-gpus 1 --config-file projects/ViTDet2/configs/V3Det/cascade_mask_rcnn_vitdet_b_4x.py "train.init_checkpoint=eva_o365.pth" "train.output_dir=output2/v3det1280"