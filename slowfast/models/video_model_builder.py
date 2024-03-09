# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Video models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import slowfast.utils.logging as logging
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.utils import validate_checkpoint_wrapper_import

import math
from functools import partial

from . import head_helper, operators, resnet_helper, stem_helper  # noqa
from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

logger = logging.get_logger(__name__)

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
    "r3d_shallow": [
        [[3]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
    "r2plus1d_shallow": [
        [[1]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels. not used
        [[3]],  # res3 temporal kernels. not used
        [[3]],  # res4 temporal kernels. not used
        [[3]],  # res5 temporal kernels. not used
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
    "r3d_shallow": [[1, 1, 1]],
    "r2plus1d_shallow": [[1, 1, 1]],
}


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._MODEL_STAGE_CHANNELS = {
            18: [
                cfg.RESNET.WIDTH_PER_GROUP,
                cfg.RESNET.WIDTH_PER_GROUP * 2,
                cfg.RESNET.WIDTH_PER_GROUP * 4,
                cfg.RESNET.WIDTH_PER_GROUP * 8,
            ],
            50: [
                cfg.RESNET.WIDTH_PER_GROUP * 4,
                cfg.RESNET.WIDTH_PER_GROUP * 8,
                cfg.RESNET.WIDTH_PER_GROUP * 16,
                cfg.RESNET.WIDTH_PER_GROUP * 32,
            ],
            101: [
                cfg.RESNET.WIDTH_PER_GROUP * 4,
                cfg.RESNET.WIDTH_PER_GROUP * 8,
                cfg.RESNET.WIDTH_PER_GROUP * 16,
                cfg.RESNET.WIDTH_PER_GROUP * 32,
            ],
        }
        self._construct_network(cfg)
        init_helper.init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )
        self.predictor_type = cfg.CONTRASTIVE.PREDICTOR_TYPE

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        self.cfg = cfg
        if cfg.MODEL.ARCH in ["r2plus1d_shallow"]:
            _MODEL_STAGE_DEPTH[18] = (1, 1, 1, 1)

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        stage_channels = self._MODEL_STAGE_CHANNELS[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
            stem_func_name=cfg.RESNET.STEM_FUNC,
        )

        s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[stage_channels[0]],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        # Based on profiling data of activation size, s1 and s2 have the activation sizes
        # that are 4X larger than the second largest. Therefore, checkpointing them gives
        # best memory savings. Further tuning is possible for better memory saving and tradeoffs
        # with recomputing FLOPs.
        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)
            self.s1 = checkpoint_wrapper(s1)
            self.s2 = checkpoint_wrapper(s2)
        else:
            self.s1 = s1
            self.s2 = s2
        if cfg.MODEL.ARCH not in ["r3d_shallow", "r2plus1d_shallow"]:
            for pathway in range(self.num_pathways):
                pool = nn.MaxPool3d(
                    kernel_size=pool_size[pathway],
                    stride=pool_size[pathway],
                    padding=[0, 0, 0],
                )
                self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[stage_channels[0]],
            dim_out=[stage_channels[1]],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[stage_channels[1]],
            dim_out=[stage_channels[2]],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[stage_channels[2]],
            dim_out=[stage_channels[3]],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        if cfg.MODEL.ARCH in ["r3d_shallow"] and cfg.MODEL.MODIFY_LAST:
            self.s5.pathway0_res0.branch2 = nn.Conv3d(
                256,
                512,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(2, 1, 1),
                dilation=(2, 1, 1),
                bias=False,
            )

            self.s5.pathway0_res0.branch1 = nn.Conv3d(256,
                                                      512,
                                                      kernel_size=(1, 1, 1),
                                                      stride=(1, 2, 2),
                                                      bias=False)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[stage_channels[-1]],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
            )

        else:
            if cfg.MODEL.ARCH not in ["r3d_shallow", "r2plus1d_shallow"]:
                timedownscale = 1
            else:
                if cfg.MODEL.MODIFY_LAST:
                    timedownscale = 4
                else:
                    timedownscale = 8
            downscale = (32 if cfg.MODEL.ARCH
                         not in ["r3d_shallow", "r2plus1d_shallow"] else 16)
            self.head = head_helper.ResNetBasicHead(
                dim_in=[stage_channels[-1]],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None] if cfg.MULTIGRID.SHORT_CYCLE
                or cfg.MODEL.MODEL_NAME == "ContrastiveModel" else [[
                    cfg.DATA.NUM_FRAMES // timedownscale // pool_size[0][0],
                    cfg.DATA.TRAIN_CROP_SIZE // downscale // pool_size[0][1],
                    cfg.DATA.TRAIN_CROP_SIZE // downscale // pool_size[0][2],
                ]],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
            )

    def forward(self, x, infos=None, bboxes=None):
        extract_flag = self.cfg.CONTRASTIVE.EXTRACT_TYPE == 'middle'
        x = x[:]  # avoid pass by reference
        x = self.s1(x)
        x = self.s2(x)
        if self.cfg.MODEL.ARCH not in ["r3d_shallow", "r2plus1d_shallow"]:
            y = [
            ]  # Don't modify x list in place due to activation checkpoint.
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                y.append(pool(x[pathway]))
        else:
            y = x
        x = self.s3(y)
        x = self.s4(x)
        x = self.s5(x, extract=extract_flag)
        if self.cfg.CONTRASTIVE.EXTRACT_TYPE == "middle":
            feat_middle = F.adaptive_avg_pool3d(x[0], (1, 1, 1)).view(
                x[0].shape[0], -1)
        if self.enable_detection:
            x_out = self.head(x, bboxes)
        else:
            x_out = self.head(x, infos)
        if self.cfg.CONTRASTIVE.EXTRACT_TYPE == "maps":
            feat_middle = F.adaptive_avg_pool3d(x[0], (1, 1, 1)).view(
                x[0].shape[0], -1)
            x_out[0] = feat_middle
        if self.cfg.CONTRASTIVE.PREDICTOR_TYPE in ["MemoryRecon"]:
            return x_out, x
        else:
            return x_out
