#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Multi-view test a video classification model."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

import json
import os
import pickle
import seaborn

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    features = None
    for cur_iter, (inputs, labels, video_idx, time,
                   meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list, )):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list, )):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()
        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (ori_boxes.detach().cpu()
                         if cfg.NUM_GPUS else ori_boxes.detach())
            metadata = (metadata.detach().cpu()
                        if cfg.NUM_GPUS else metadata.detach())

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes),
                                      dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (model.module.train_labels if hasattr(
                model, "module") else model.train_labels)
            yd, yi, q_knn = model(inputs, video_idx, time)
            # yd: bs * 200
            # yi: bs * 200
            # q_knn: bs * 512, normed feature
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            # Perform the forward pass.
            preds = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()

        if not cfg.VIS_MASK.ENABLE:
            # Update and log stats.
            test_meter.update_stats(preds.detach(), labels.detach(),
                                    video_idx.detach())
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            np.save(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "test_features.npy"),
                features,
            )
            np.save(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix, "test_preds.npy"),
                all_preds,
            )
            np.save(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "test_labels.npy"),
                all_labels,
            )

            logger.info("Successfully saved prediction results to {}".format(
                cfg.OUTPUT_DIR))
    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(model,
                                                cfg,
                                                use_train_input=False)

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model,
                                cfg,
                                use_train_input=False,
                                print_model=False)
        if (cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                and cfg.CONTRASTIVE.KNN_ON):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert (test_loader.dataset.num_videos %
                    (cfg.TEST.NUM_ENSEMBLE_VIEWS *
                     cfg.TEST.NUM_SPATIAL_CROPS) == 0)
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                test_loader.dataset.num_videos //
                (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES if not cfg.TASK == "ssl" else
                cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
                cfg.NUM_GPUS * cfg.NUM_SHARDS):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()

    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        logger.info(
            "Finalized testing with {} temporal clips and {} spatial crops".
            format(view, cfg.TEST.NUM_SPATIAL_CROPS))
        result_string_views += "_{}a{}" "".format(view,
                                                  test_meter.stats["top1_acc"])

        result_string = (
            "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
            "".format(
                params / 1e6,
                flops,
                view,
                test_meter.stats["top1_acc"],
                test_meter.stats["top5_acc"],
                misc.gpu_mem_usage(),
                flops,
            ))

        logger.info("{}".format(result_string))
    logger.info("{}".format(result_string_views))
    return result_string + " \n " + result_string_views


@torch.no_grad()
def extract_feat(
    test_loader,
    model,
    test_meter,
    cfg,
    writer=None,
    mode="test",
    feature_prefix="",
):
    model.eval()
    test_meter.iter_tic()
    features = None
    test_labels = None
    video_idxs = None

    for cur_iter, (inputs, labels, video_idx, time,
                   meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list, )):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list, )):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()
        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (ori_boxes.detach().cpu()
                         if cfg.NUM_GPUS else ori_boxes.detach())
            metadata = (metadata.detach().cpu()
                        if cfg.NUM_GPUS else metadata.detach())

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes),
                                      dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (model.module.train_labels if hasattr(
                model, "module") else model.train_labels)
            # yd: bs * 200, yi: bs * 200, q_knn: bs * 512, normed feature
            yd, yi, q_knn = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)

            feat_cur = q_knn.cpu().detach().numpy()
            labels_cur = labels.cpu().detach().numpy()
            idxs_cur = video_idx.cpu().detach().numpy()
            features = (feat_cur if features is None else np.append(
                features, feat_cur))
            test_labels = (labels_cur if test_labels is None else np.append(
                test_labels, labels_cur))
            video_idxs = (idxs_cur if features is None else np.append(
                video_idxs, idxs_cur))
        else:
            # Perform the forward pass.
            preds = model(inputs)

        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        if not cfg.VIS_MASK.ENABLE:
            # Update and log stats.
            test_meter.update_stats(preds.detach(), labels.detach(),
                                    video_idx.detach())
        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()
    np.save(
        os.path.join(cfg.OUTPUT_DIR, feature_prefix, f"{mode}_features.npy"),
        features.reshape(test_loader.dataset.num_videos, -1),
    )
    np.save(
        os.path.join(cfg.OUTPUT_DIR, feature_prefix, f"{mode}_labels.npy"),
        test_labels,
    )
    logger.info(f"Successfully saved {mode} features to {cfg.OUTPUT_DIR}")
    test_meter.finalize_metrics()
    return test_meter, features, test_labels


def test_retrieval(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    model_name = cfg.TEST.CHECKPOINT_FILE_PATH.split('/')[-1].replace(
        '.pyth', '')
    feature_prefix = cfg.CONTRASTIVE.EXTRACT_TYPE + '_' + cfg.DATA.PATH_TO_DATA_DIR.split(
        "/")[-1] + f'_{model_name}'
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, feature_prefix)):
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, feature_prefix))

    logger.info(
        f"Saving the extracted features to {os.path.join(cfg.OUTPUT_DIR, feature_prefix)}"
    )
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        flops, params = 0.0, 0.0
        #if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        #    model.eval()
        #    flops, params = misc.log_model_info(
        #        model, cfg, use_train_input=False
        #    )
        #if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        #    misc.log_model_info(model, cfg, use_train_input=False)
        if (cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                and cfg.CONTRASTIVE.KNN_ON):
            train_loader = loader.construct_loader(cfg, "retrieval_train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)
        # for name, param in model.named_parameters():
        #     print(f"{name}, {torch.norm(param)}")
        extract_new = False
        if (os.path.exists(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "test_features.npy")) and not extract_new):
            print(
                f"Test features extracted. Loaded {os.path.join(cfg.OUTPUT_DIR, feature_prefix, 'test_labels.npy')}."
            )
            test_label = np.load(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "test_labels.npy"))
            test_feature = np.load(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "test_features.npy")).reshape(
                                 test_label.shape[0], -1)
        else:
            print(f"Not exsit Test features. Extracting...")
            # Create video testing loaders.
            test_loader = loader.construct_loader(cfg, "retrieval_val")
            logger.info(
                "Retrieval test: Testing model for {} iterations".format(
                    len(test_loader)))
            test_meter = TestMeter(
                test_loader.dataset.num_videos,
                1,
                cfg.MODEL.NUM_CLASSES if not cfg.TASK == "ssl" else
                cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )
            # Set up writer for logging to Tensorboard format.
            if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
                    cfg.NUM_GPUS * cfg.NUM_SHARDS):
                writer = tb.TensorboardWriter(cfg)
            else:
                writer = None
            # # Perform multi-view test on the entire dataset.
            test_meter, test_feature, test_label = extract_feat(
                test_loader,
                model,
                test_meter,
                cfg,
                writer,
                "test",
                feature_prefix,
            )
            test_meters.append(test_meter)
            if writer is not None:
                writer.close()

        # ------------------------
        if (os.path.exists(
                os.path.join(
                    cfg.OUTPUT_DIR,
                    feature_prefix,
                    f"test{num_view*cfg.TEST.NUM_SPATIAL_CROPS}_features.npy",
                )) and not extract_new):
            print(
                f"{num_view*cfg.TEST.NUM_SPATIAL_CROPS} Test features extracted. Loaded {os.path.join(cfg.OUTPUT_DIR, feature_prefix, f'test{num_view*cfg.TEST.NUM_SPATIAL_CROPS}_labels.npy')}."
            )
            test_multi_label = np.load(
                os.path.join(
                    cfg.OUTPUT_DIR, feature_prefix,
                    f"test{num_view*cfg.TEST.NUM_SPATIAL_CROPS}_labels.npy"))
            test_multi_feature = np.load(
                os.path.join(
                    cfg.OUTPUT_DIR,
                    feature_prefix,
                    f"test{num_view}_features.npy",
                )).reshape(test_multi_label.shape[0], -1)

        else:
            print(
                f"Not exsit Test {num_view*cfg.TEST.NUM_SPATIAL_CROPS} features. Extracting..."
            )
            # Create video testing loaders.
            test_multi_loader = loader.construct_loader(cfg, "test")
            logger.info(
                f"Retrieval test {num_view*cfg.TEST.NUM_SPATIAL_CROPS}: Testing model for {len(test_multi_loader)} iterations"
            )
            test_meter = TestMeter(
                test_multi_loader.dataset.num_videos //
                (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES if not cfg.TASK == "ssl" else
                cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_multi_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )
            # Set up writer for logging to Tensorboard format.
            if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
                    cfg.NUM_GPUS * cfg.NUM_SHARDS):
                writer = tb.TensorboardWriter(cfg)
            else:
                writer = None
            # # Perform multi-view test on the entire dataset.
            test_meter, test_multi_feature, test_multi_label = extract_feat(
                test_multi_loader,
                model,
                test_meter,
                cfg,
                writer,
                f"test{num_view*cfg.TEST.NUM_SPATIAL_CROPS}",
                feature_prefix,
            )
            test_meters.append(test_meter)
            if writer is not None:
                writer.close()
        # ------------------------
        if (os.path.exists(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "train_features.npy")) and not extract_new):
            print(
                f"Train features extracted. Loaded {os.path.join(cfg.OUTPUT_DIR, feature_prefix, 'train_labels.npy')}."
            )
            train_label = np.load(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "train_labels.npy"))
            train_feature = np.load(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             "train_features.npy")).reshape(
                                 train_label.shape[0], -1)
        else:
            print(f"Not exsit Train features. Extracting...")
            # Create video testing loaders.
            train_loader = loader.construct_loader(cfg, "retrieval_train")
            logger.info(
                "Retrieval train: Testing model for {} iterations".format(
                    len(train_loader)))
            test_meter = TestMeter(
                train_loader.dataset.num_videos,
                1,
                cfg.MODEL.NUM_CLASSES if not cfg.TASK == "ssl" else
                cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(train_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )
            # Set up writer for logging to Tensorboard format.
            if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
                    cfg.NUM_GPUS * cfg.NUM_SHARDS):
                writer = tb.TensorboardWriter(cfg)
            else:
                writer = None
            # # Perform multi-view test on the entire dataset.
            test_meter, train_feature, train_label = extract_feat(
                train_loader,
                model,
                test_meter,
                cfg,
                writer,
                "train",
                feature_prefix,
            )
            test_meters.append(test_meter)

            if writer is not None:
                writer.close()

    # ---------Single Clip Retrieval Results------------
    ks = [1, 5, 10, 20, 50]
    NN_acc = []
    train_label = torch.from_numpy(train_label).cuda()
    test_label = torch.from_numpy(test_label).cuda()
    test_feature = (torch.from_numpy(test_feature).cuda().reshape(
        test_label.shape[0], -1))
    train_feature = (torch.from_numpy(train_feature).cuda().reshape(
        train_label.shape[0], -1))
    sim = torch.matmul(
        F.normalize(test_feature, dim=-1),
        F.normalize(train_feature, dim=-1).transpose(0, 1),
    )
    print(test_feature.shape, train_feature.shape)
    np.save(
        os.path.join(cfg.OUTPUT_DIR, feature_prefix, "retrieval_sim.npy"),
        sim.cpu().detach().numpy(),
    )
    result_string = "Single Clip Retrieval: "
    for k in ks:
        topkval, topkidx = torch.topk(sim, k, dim=1)
        acc = (torch.any(train_label[topkidx] == test_label.unsqueeze(1),
                         dim=1).float().mean().item())
        NN_acc.append(acc)
        result_string += "Top%d: %.4f " % (k, acc)
    logger.info(result_string)
    # ---------Multi Clip Retrieval Results------------
    num_retrieval_view = (cfg.TEST.NUM_TEMPORAL_CLIPS[0] *
                          cfg.TEST.NUM_SPATIAL_CROPS)
    test_multi_feature = (torch.from_numpy(test_multi_feature).cuda().reshape(
        test_multi_label.shape[0] // num_retrieval_view,
        num_retrieval_view,
        -1,
    ))
    logger.info(f"Feature shape is {test_multi_feature.shape}")
    print(test_feature.shape)
    assert test_multi_feature.shape[0] == test_feature.shape[0]
    #test_multi_feature_mean = F.normalize(test_multi_feature, dim=-1).mean(1)
    test_multi_feature_mean = test_multi_feature.mean(1)

    # centering
    #test_multi_feature_mean = (
    #    test_multi_feature_mean
    #    - test_multi_feature_mean.mean(dim=0, keepdim=True)
    #)

    # normalize
    sim_multi = torch.matmul(
        F.normalize(test_multi_feature_mean, dim=-1),
        F.normalize(train_feature, dim=-1).transpose(0, 1),
    )
    np.save(
        os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                     "retrieval_sim_multi.npy"),
        sim_multi.cpu().detach().numpy(),
    )
    result_string = f"Multi {num_view} Clip Retrieval: "
    for k in ks:
        topkval, topkidx = torch.topk(sim_multi, k, dim=1)
        acc = (torch.any(train_label[topkidx] == test_label.unsqueeze(1),
                         dim=1).float().mean().item())
        NN_acc.append(acc)
        result_string += "Top%d: %.4f " % (k, acc)
    logger.info(result_string)

    # Similarity results
    mean_feature = F.normalize(torch.mean(test_multi_feature, dim=1), dim=-1)
    inter_sim_matrix = 1 - torch.matmul(mean_feature,
                                        mean_feature.transpose(0, 1))
    inter_sim_score = inter_sim_matrix.mean()
    inner_sim_matrix = 1 - torch.einsum(
        "bcd,bed->bce",
        F.normalize(test_multi_feature, dim=-1),
        F.normalize(test_multi_feature, dim=-1),
    )
    inner_sim_score = inner_sim_matrix.mean()
    logger.info(f"Dissimilarity of mean feature (30 views): {inter_sim_score}")
    logger.info(f"Dissimilarity of 30 views of one video: {inner_sim_score}")

    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)
    result_string = ""
    for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        logger.info(
            "Finalized testing with {} temporal clips and {} spatial crops".
            format(view, cfg.TEST.NUM_SPATIAL_CROPS))
        result_string_views += "_{}a{}" "".format(view,
                                                  test_meter.stats["top1_acc"])

        result_string = (
            "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
            "".format(
                params / 1e6,
                flops,
                view,
                test_meter.stats["top1_acc"],
                test_meter.stats["top5_acc"],
                misc.gpu_mem_usage(),
                flops,
            ))

        logger.info("{}".format(result_string))
    logger.info("{}".format(result_string_views))
    return result_string + " \n " + result_string_views


@torch.no_grad()
def extract_middle(
    test_loader,
    model,
    test_meter,
    cfg,
    writer=None,
    mode="test",
    feature_prefix="",
):
    model.eval()
    test_meter.iter_tic()
    features = {}
    test_labels = None
    video_idxs = None

    for cur_iter, (inputs, labels, video_idx, time,
                   meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list, )):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list, )):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()
        if not (cfg.TASK == "ssl"
                and cfg.MODEL.MODEL_NAME == "ContrastiveModel"):
            raise RuntimeError(f"Only can be used in ContrastiveModel and ssl")
        labels_cur = labels.cpu().detach().numpy()
        idxs_cur = video_idx.cpu().detach().numpy()
        test_labels = (labels_cur if test_labels is None else np.append(
            test_labels, labels_cur))
        video_idxs = (idxs_cur if features is None else np.append(
            video_idxs, idxs_cur))
        # yd: bs * 200, yi: bs * 200, q_knn: bs * 512, normed feature
        vis_feature = model(inputs, video_idx, time)
        for key in vis_feature:
            feat_cur = vis_feature[key].cpu().detach().numpy()
            features[key] = (feat_cur if key not in features else np.append(
                features[key], feat_cur))

        if cur_iter % 10 == 0:
            test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()
    for key in features:
        np.save(
            os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                         f"{mode}_{key}_features.npy"),
            features[key].reshape(test_loader.dataset.num_videos, -1),
        )
    np.save(
        os.path.join(cfg.OUTPUT_DIR, feature_prefix, f"{mode}_labels.npy"),
        test_labels,
    )
    logger.info(f"Successfully saved {mode} features to {cfg.OUTPUT_DIR}")
    test_meter.finalize_metrics()
    return test_meter, features, test_labels


def test_middle(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    model_name = cfg.TEST.CHECKPOINT_FILE_PATH.split('/')[-1].replace(
        '.pyth', '')
    feature_prefix = cfg.CONTRASTIVE.EXTRACT_TYPE + '_' + cfg.DATA.PATH_TO_DATA_DIR.split(
        "/")[-1] + f'_{model_name}'
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, feature_prefix)):
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, feature_prefix))

    logger.info(
        f"Saving the extracted features to {os.path.join(cfg.OUTPUT_DIR, feature_prefix)}"
    )
    cfg.TEST.NUM_SPATIAL_CROPS = 1
    cfg.TEST.NUM_TEMPORAL_CLIPS = [10]
    extract_new_vis = False
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:
        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        flops, params = 0.0, 0.0
        #if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        #    model.eval()
        #    flops, params = misc.log_model_info(
        #        model, cfg, use_train_input=False
        #    )
        #if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        #    misc.log_model_info(model, cfg, use_train_input=False)
        if (cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                and cfg.CONTRASTIVE.KNN_ON):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)
        # for name, param in model.named_parameters():
        #     print(f"{name}, {torch.norm(param)}")
        vis_keys = [
            "vis_source_mem_sim",
            "vis_target_mem_sim",
            "vis_cross_mem_sim",
        ]
        if os.path.exists(
                os.path.join(
                    cfg.OUTPUT_DIR,
                    feature_prefix,
                    f"test{num_view}_{vis_keys[2]}_features.npy",
                )) and not extract_new_vis:
            test_multi_label = np.load(
                os.path.join(cfg.OUTPUT_DIR, feature_prefix,
                             f"test{num_view}_labels.npy"))
            test_multi_features = {}
            for key in vis_keys:
                feature_path = os.path.join(
                    cfg.OUTPUT_DIR,
                    feature_prefix,
                    f"test{num_view}_{key}_features.npy",
                )
                if not os.path.exists(feature_path):
                    continue
                test_multi_feature = np.load(feature_path).reshape(
                    test_multi_label.shape[0], -1)
                test_multi_features[key] = test_multi_feature
                print(
                    f"{num_view} Test ({key}) features {feature_path} extracted. Loaded {cfg.OUTPUT_DIR, feature_prefix, f'test{num_view}_labels.npy'}."
                )

        else:
            print(f"Not exsit Test {num_view} features. Extracting...")
            # Create video testing loaders.
            test_multi_loader = loader.construct_loader(cfg, "test")
            logger.info(
                f"Retrieval test {num_view}: Testing model for {len(test_multi_loader)} iterations"
            )
            test_meter = TestMeter(
                test_multi_loader.dataset.num_videos //
                (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES if not cfg.TASK == "ssl" else
                cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_multi_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )
            # Set up writer for logging to Tensorboard format.
            if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
                    cfg.NUM_GPUS * cfg.NUM_SHARDS):
                writer = tb.TensorboardWriter(cfg)
            else:
                writer = None
            # # Perform multi-view test on the entire dataset.
            test_meter, test_multi_features, test_multi_label = extract_middle(
                test_multi_loader,
                model,
                test_meter,
                cfg,
                writer,
                f"test{num_view}",
                feature_prefix,
            )
            test_meters.append(test_meter)
            if writer is not None:
                writer.close()

    # ---------Vis: topk index------------
    num_retrieval_view = (cfg.TEST.NUM_TEMPORAL_CLIPS[0] *
                          cfg.TEST.NUM_SPATIAL_CROPS)
    fig = plt.figure()
    test_multi_label = torch.from_numpy(test_multi_label)
    cfg.DATA.MEAN = [0, 0, 0]
    cfg.DATA.STD = [1, 1, 1]
    test_multi_loader = loader.construct_loader(cfg, "test")
    names = json.load(open('tools/classInd.json', 'r'))
    class_dict = {}
    for name in names:
        class_dict[int(names[name]) - 1] = f'{name}'
    names = class_dict
    for key in test_multi_features:
        features = (torch.from_numpy(test_multi_features[key]).cuda()).reshape(
            test_multi_label.shape[0], -1)

        topk = 20
        count = {}
        topk_slots = 50
        topk_retrieval = 10
        vis_classes = [22, 23, 61, 66, 69, 46]
        #vis_classes = [23,24,32,19,62,67,70,47]
        #vis_classes = [22, 23, 31,18,61,66,69,46,0, 1, 50, 30, 20, 51, 10, 96]

        logger.info(f"-------Dataset Vise {key}-------")
        features_part = None
        if key.find("cross_mem") >= 0 and cfg.OUTPUT_DIR.find("h8") >= 0:
            features = F.softmax(features.reshape(features.shape[0], 8, -1) *
                                 cfg.MEMORY.RADIUS,
                                 dim=-1).reshape(features.shape[0], -1)
        else:
            if key.find('cross_mem') >= 0:
                features = F.softmax(features * cfg.MEMORY.RADIUS, dim=-1)
            elif key.find('source_mem') >= 0:
                features = F.softmax(features / cfg.MEMORY.DINO_TEMP_S, dim=-1)
            else:
                features = F.softmax(features / cfg.MEMORY.DINO_TEMP_T, dim=-1)
        # return topk samples for each slot, [topk_sample, slot]
        slot_topk = torch.topk(features, topk_retrieval, dim=0)
        choose_slots = torch.topk(torch.sum(slot_topk[0], dim=0),
                                  topk_retrieval)[1]
        choose_samples_for_slot = slot_topk[1][:, choose_slots]
        choose_samples_scores_for_slot = slot_topk[0][:, choose_slots]
        for slot_idx in range(choose_samples_for_slot.shape[1]):
            samples = choose_samples_for_slot[:, slot_idx]
            for index in range(samples.shape[0]):
                index_sample = int(samples[index])
                frames, label, index_return, time_idx, _ = test_multi_loader.dataset.__getitem__(
                    index_sample)
                # 3, 16, 128, 128
                frames = frames[0].cpu()
                frames = [
                    ToPILImage()(frames[:, i])
                    for i in range(0, frames.shape[1], 5)
                ]
                video_path = f"{cfg.OUTPUT_DIR}/vis_{num_retrieval_view}view_{key}/slot_index{choose_slots[slot_idx]}_score{choose_samples_scores_for_slot[index, slot_idx]}_{names[label]}"
                if not os.path.exists(video_path):
                    os.makedirs(video_path)
                print(f"saving index {video_path}")
                for id, im in enumerate(frames):
                    im.save(f"{video_path}/{id}.jpg")
        #-------------------choose all v_classes samples and give choose_slots--------------
        for cls_id in vis_classes:
            class_bool = test_multi_label == cls_id
            feature_cls = features[class_bool]
            if features_part is None:
                features_part = feature_cls
            else:
                features_part = torch.cat((features_part, feature_cls), dim=0)

        #sample_topk = torch.topk(features_part, 3,
        #                         dim=1)  # return topk slot for each sample, [num, topk_slot]
        scores_classes, idxs_classes = torch.topk(features_part, topk, dim=-1)
        logger.info(f"{scores_classes}, {idxs_classes}")
        count_global = []
        for slot_id in range(features_part.shape[1]):
            count_global.append(
                (torch.sum(idxs_classes.reshape(-1) == slot_id)))
        count_global = torch.tensor(count_global)
        choose_idxs = torch.topk(count_global, topk_slots)[1]
        logger.info(torch.topk(count_global, topk_slots))

        # plt.bar(range(features.shape[1]), np.array(count_global), width=2)
        # plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"{key}_global.png"), dpi=300)
        # fig.clear()

        # use choose_slots vis class
        probs = None
        for cls_id in vis_classes:
            class_bool = test_multi_label == cls_id
            # num, 4096
            feature_cls = features[class_bool]
            logger.info(
                f"Number of samples in Class {cls_id} is {feature_cls.shape[0]}"
            )
            scores_cls_topk, cls_topk_index = torch.topk(feature_cls,
                                                         topk,
                                                         dim=-1)
            # logger.info(f"{scores_cls_topk}, {cls_topk_index}")
            count[cls_id] = []
            # for slot_id in range(feature_cls.shape[1]):
            #     count[cls_id].append(
            #         (torch.sum(cls_topk_index.reshape(-1) == slot_id))
            #     )
            for slot_id in choose_idxs:
                count[cls_id].append(
                    (torch.sum(cls_topk_index.reshape(-1) == slot_id)))
            count[cls_id] = torch.tensor(count[cls_id])
            if probs is not None:
                probs = torch.cat([probs, count[cls_id]], dim=0)
            else:
                probs = count[cls_id]
            # logger.info(torch.topk(count[cls_id], topk))
            # ------- vis bar --------
            # plt.bar(
            #     range(feature_cls.shape[1]), np.array(count[cls_id]), width=2
            # )
            # plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"{key}_{cls_id}.png"))
            # fig.clear()
        class_number = len(vis_classes)
        probs = probs.reshape(class_number, -1)
        print('before mask', probs.shape)
        class_times = torch.sum(probs, dim=0)
        class_times[class_times == 0] = 1
        appear_mask = class_times > 0  # 4096
        probs = probs[appear_mask.repeat(class_number,
                                         1)].reshape(class_number, -1)
        #probs =  probs.reshape(class_number, -1)
        print('after mask', probs.shape)
        norm_num = torch.sum(probs, dim=0)

        print(norm_num, torch.sum(norm_num))
        norm_num[norm_num == 0] = 1
        probs_slot = probs / norm_num
        #probs_slot = F.softmax(probs_slot.float() * 1000, dim=-1)

        norm_num = torch.sum(probs, dim=1)
        print(norm_num, torch.sum(norm_num))
        norm_num[norm_num == 0] = 1
        probs_class = probs / norm_num.reshape(-1, 1)
        #probs_class = F.softmax(probs_class.float() * 1000, dim=-1)

        # vis class
        xticklabels = [f'{id}' for id in choose_idxs[appear_mask]]
        #xticklabels = [f'{id}' for id in choose_idxs]

        yticklabels = [f'{names[id]}' for id in vis_classes]
        ax = seaborn.heatmap(np.array(probs_class),
                             mask=None,
                             vmin=None,
                             vmax=None,
                             center=None,
                             annot=False,
                             cmap=plt.get_cmap('viridis'),
                             linewidths=.5,
                             cbar=True,
                             square=False,
                             xticklabels=xticklabels,
                             yticklabels=yticklabels)
        label_x = ax.get_xticklabels()
        plt.setp(label_x, rotation=90, horizontalalignment='right', fontsize=5)
        label_y = ax.get_yticklabels()
        plt.setp(label_y, rotation=45, horizontalalignment='right', fontsize=8)
        plt.savefig(os.path.join(
            cfg.OUTPUT_DIR,
            f"{key}_{vis_classes}_{topk_slots}slot_{topk}act_prob_normclass.png"
        ),
                    dpi=100)
        fig.clear()

        # vis slot
        ax = seaborn.heatmap(np.array(probs_slot),
                             mask=None,
                             vmin=0,
                             vmax=1,
                             center=None,
                             annot=False,
                             cmap=plt.get_cmap('viridis'),
                             linewidths=.5,
                             cbar=True,
                             square=False,
                             xticklabels=xticklabels,
                             yticklabels=yticklabels)
        label_x = ax.get_xticklabels()
        plt.setp(label_x, rotation=90, horizontalalignment='right', fontsize=5)
        label_y = ax.get_yticklabels()
        plt.setp(label_y, rotation=45, horizontalalignment='right', fontsize=8)
        plt.savefig(os.path.join(
            cfg.OUTPUT_DIR,
            f"{key}_{vis_classes}_{topk_slots}slot_{topk}act_prob_normslot.png"
        ),
                    dpi=100)
        fig.clear()

    result_string = ""
    return result_string
