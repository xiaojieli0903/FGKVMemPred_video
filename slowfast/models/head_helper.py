import numpy as np
import torch
import torch.nn as nn
from detectron2.layers import ROIAlign

import slowfast.utils.logging as logging
from slowfast.models.batchnorm_helper import NaiveSyncBatchNorm1d as NaiveSyncBatchNorm1d
from slowfast.models.memory_recon import MemoryRecon

logger = logging.get_logger(__name__)


class MLPHead(nn.Module):

    def __init__(
        self,
        dim_in,
        dim_out,
        mlp_dim,
        num_layers,
        bn_on=False,
        bias=True,
        flatten=False,
        xavier_init=True,
        bn_sync_num=1,
        global_sync=False,
    ):
        super(MLPHead, self).__init__()
        self.flatten = flatten
        b = False if bn_on else bias
        # assert bn_on or bn_sync_num=1
        mlp_layers = [nn.Linear(dim_in, mlp_dim, bias=b)]
        mlp_layers[-1].xavier_init = xavier_init
        for i in range(1, num_layers):
            if bn_on:
                if global_sync or bn_sync_num > 1:
                    mlp_layers.append(
                        NaiveSyncBatchNorm1d(
                            num_sync_devices=bn_sync_num,
                            global_sync=global_sync,
                            num_features=mlp_dim,
                        ))
                else:
                    mlp_layers.append(nn.BatchNorm1d(num_features=mlp_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if i == num_layers - 1:
                d = dim_out
                b = bias
            else:
                d = mlp_dim
            mlp_layers.append(nn.Linear(mlp_dim, d, bias=b))
            mlp_layers[-1].xavier_init = xavier_init
        self.projection = nn.Sequential(*mlp_layers)

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute((0, 2, 3, 4, 1))
        if self.flatten:
            x = x.reshape(-1, x.shape[-1])

        return self.projection(x)


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        detach_final_fc=False,
        cfg=None,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            detach_final_fc (bool): if True, detach the fc layer from the
                gradient graph. By doing so, only the final fc layer will be
                trained.
            cfg (struct): The config for the current experiment.
        """
        super(ResNetBasicHead, self).__init__()
        assert (len({len(pool_size), len(dim_in)
                     }) == 1), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.detach_final_fc = detach_final_fc
        self.cfg = cfg
        self.local_projection_modules = []
        self.predictors = nn.ModuleList()
        self.l2norm_feats = False
        self.predictor_type = cfg.CONTRASTIVE.PREDICTOR_TYPE

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        if cfg.CONTRASTIVE.NUM_MLP_LAYERS == 1:
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        else:
            self.projection = MLPHead(
                sum(dim_in),
                num_classes,
                cfg.CONTRASTIVE.MLP_DIM,
                cfg.CONTRASTIVE.NUM_MLP_LAYERS,
                bn_on=cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=cfg.BN.NUM_SYNC_DEVICES
                if cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(cfg.CONTRASTIVE.BN_SYNC_MLP
                             and cfg.BN.GLOBAL_SYNC),
            )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError("{} is not supported as an activation"
                                      "function.".format(act_func))

        if cfg.CONTRASTIVE.PREDICTOR_DEPTHS:
            d_in = num_classes
            for i, n_layers in enumerate(cfg.CONTRASTIVE.PREDICTOR_DEPTHS):
                if cfg.CONTRASTIVE.PREDICTOR_TYPE in ["MemoryRecon"]:
                    local_mlp = MemoryRecon(dim_input=d_in,
                                            dim_output=num_classes,
                                            cfg=cfg)
                    self.predictors.append(local_mlp)
                elif cfg.CONTRASTIVE.TYPE == "dino":
                    local_mlp = nn.utils.weight_norm(
                        nn.Linear(d_in, cfg.CONTRASTIVE.MLP_DIM, bias=False))
                    local_mlp.weight_g.data.fill_(1)
                    local_mlp.weight_g.requires_grad = False  # norm last layer
                    self.predictors.append(local_mlp)
                elif cfg.CONTRASTIVE.TYPE == "mlp_v1":
                    local_mlp = nn.utils.weight_norm(
                        nn.Linear(d_in, cfg.CONTRASTIVE.MLP_DIM, bias=False))
                    self.predictors.append(local_mlp)
                    local_mlp_1 = MLPHead(
                        cfg.CONTRASTIVE.MLP_DIM,
                        num_classes,
                        num_classes,
                        n_layers,
                        bn_on=cfg.CONTRASTIVE.BN_MLP,
                        flatten=False,
                        bn_sync_num=cfg.BN.NUM_SYNC_DEVICES
                        if cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                        global_sync=(cfg.CONTRASTIVE.BN_SYNC_MLP
                                     and cfg.BN.GLOBAL_SYNC),
                    )
                    self.predictors.append(local_mlp_1)
                else:
                    local_mlp = MLPHead(
                        d_in,
                        num_classes,
                        cfg.CONTRASTIVE.MLP_DIM,
                        n_layers,
                        bn_on=cfg.CONTRASTIVE.BN_MLP,
                        flatten=False,
                        bn_sync_num=cfg.BN.NUM_SYNC_DEVICES
                        if cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                        global_sync=(cfg.CONTRASTIVE.BN_SYNC_MLP
                                     and cfg.BN.GLOBAL_SYNC),
                    )
                    self.predictors.append(local_mlp)

    def forward(self, inputs, infos=None):
        # inputs: [[32, 2048, 8, 7, 7]]
        assert (len(inputs) == self.num_pathways
                ), "Input tensor does not contain {} pathway".format(
                    self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        # pool_out: [[32, 2048, 1, 1, 1]]
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # x: torch.Size([32, 1, 1, 1, 2048])
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.detach_final_fc:
            x = x.detach()
        if self.l2norm_feats:
            x = nn.functional.normalize(x, dim=1, p=2)

        if (x.shape[1:4] == torch.Size([1, 1, 1])
                and self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"):
            x = x.view(x.shape[0], -1)

        x_proj = self.projection(x)

        time_projs = []
        if self.predictors:
            x_in = x_proj
            target, time, pred_maps = None, None, None
            if infos is not None:
                if infos[0] is not None:
                    # len(target) = num_clip - 1
                    target = infos[0]
                if len(infos) > 1 and infos[1] is not None:
                    time = infos[1]
                if len(infos) > 2 and infos[2] is not None:
                    # len(pred_maps) = num_clip - 1
                    pred_maps = infos[2]
            id_proj = 0
            for proj in self.predictors:
                if self.cfg.CONTRASTIVE.TYPE in ["dino"]:
                    time_projs.append(
                        proj(nn.functional.normalize(x_in, dim=-1, p=2)))
                elif self.cfg.CONTRASTIVE.TYPE in ["mlp_v1"]:
                    # import pdb; pdb.set_trace()
                    if id_proj == 0:
                        time_projs.append(
                            proj(nn.functional.normalize(x_in, dim=-1, p=2)))
                    else:
                        time_projs.append(proj((time_projs[-1])))
                elif self.predictor_type in ["MemoryRecon"]:
                    time_projs.append(proj(x_in, [target, inputs, pred_maps]))
                else:
                    time_projs.append(proj(x_in))
                id_proj += 1
            if len(time_projs) != 1:
                time_projs = [time_projs[-1]]
        if not self.training:
            if self.act is not None:
                x_proj = self.act(x_proj)
            # Performs fully convlutional inference.
            if x_proj.ndim == 5 and x_proj.shape[1:4] > torch.Size([1, 1, 1]):
                x_proj = x_proj.mean([1, 2, 3])

        x_proj = x_proj.view(x_proj.shape[0], -1)

        if time_projs:
            return [x_proj] + time_projs
        else:
            return x_proj
