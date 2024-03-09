#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Wrapper to train and test a video classification model."""
import torch

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_net import test, test_middle, test_retrieval
from train_net import train
from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    torch.cuda.empty_cache()
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        # Perform training.
        if cfg.TRAIN.ENABLE:
            if cfg.TRAIN.VIS_INPUT:
                cfg.DATA.MEAN = [0, 0, 0]
                cfg.DATA.STD = [1, 1, 1]
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

        # Perform multi-clip testing.
        if cfg.TEST.ENABLE:
            if cfg.TEST.VIS_INPUT:
                cfg.DATA.MEAN = [0, 0, 0]
                cfg.DATA.STD = [1, 1, 1]
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                if cfg.NUM_GPUS > 1:
                    print(f"Num of GPUS is {cfg.NUM_GPUS}, not testing.")
                    continue
                if cfg.TEST.VIS_MIDDLE:
                    launch_job(cfg=cfg,
                               init_method=args.init_method,
                               func=test_middle)
                else:
                    #cfg.DATA.SSL_COLOR_BRI_CON_SAT = [0.1, 0.1, 0.1]
                    cfg.DATA.SSL_MOCOV2_AUG = False
                    cfg.DATA.COLOR_RND_GRAYSCALE = 0.0
                    cfg.DATA.TRAIN_JITTER_SCALES = [128, 160]
                    cfg.DATA.TEST_CROP_SIZE = 112
                    cfg.DATA.TRAIN_CROP_SIZE = 112
                    cfg.CONTRASTIVE.EXTRACT_TYPE = "maps"
                    launch_job(
                        cfg=cfg,
                        init_method=args.init_method,
                        func=test_retrieval,
                    )
            else:
                if cfg.TEST.NUM_ENSEMBLE_VIEWS == -1:
                    num_view_list = [1, 3, 5, 7, 10]
                    for num_view in num_view_list:
                        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view
                        launch_job(cfg=cfg,
                                   init_method=args.init_method,
                                   func=test)
                else:
                    launch_job(cfg=cfg,
                               init_method=args.init_method,
                               func=test)

        # Perform model visualization.
        if cfg.TENSORBOARD.ENABLE and (cfg.TENSORBOARD.MODEL_VIS.ENABLE or
                                       cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE):
            cfg.TENSORBOARD.MODEL_VIS.LAYER_LIST = [
                "backbone/s2/pathway0_res1/relu"
            ]
            cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = [
                "backbone/s2/pathway0_res1/relu"
            ]
            cfg.TENSORBOARD.CLASS_NAMES_PATH = (
                "/path/to/datasets/UCF101/annotations/classInd.json")
            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

        # Run demo.
        if cfg.DEMO.ENABLE:
            demo(cfg)


if __name__ == "__main__":
    main()
