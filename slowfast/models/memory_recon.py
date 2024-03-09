import torch
import torch.nn.functional as F
from torch import nn

from slowfast.utils import distributed as du

from . import head_helper


class DINOLoss(nn.Module):
    """
    DINOLoss: Cross-entropy loss between softmax outputs of teacher and student networks.
    ref: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py#L363
    """

    def __init__(self,
                 out_dim,
                 teacher_temp=0.01,
                 student_temp=0.1,
                 center_momentum=0.9,
                 sync_center=True):
        """
        Initialize DINOLoss.

        Args:
            out_dim (int): Dimensionality of output.
            teacher_temp (float): Temperature for teacher network.
            student_temp (float): Temperature for student network.
            center_momentum (float): Momentum for center update.
            sync_center (bool): Flag to synchronize center update.
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp
        self.sync_center = sync_center

    def forward(self, student_output, teacher_output):
        """
        Compute the loss using cross-entropy between softmax outputs.

        Args:
            student_output (Tensor): Output of student network.
            teacher_output (Tensor): Output of teacher network.

        Returns:
            loss (Tensor): Computed loss.
        """
        student_out = student_output / self.student_temp
        teacher_out = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1),
                         dim=-1).mean()
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update the center used for teacher output.

        Args:
            teacher_output (Tensor): Output of teacher network.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.sync_center:
            du.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) *
                                           du.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum)


class MemoryRecon(nn.Module):
    """
    MemoryRecon: Module for memory reconstruction.
    """

    def __init__(self, dim_input=256, dim_output=256, cfg=None):
        """
        Initializes the MemoryRecon module.

        Args:
            dim_input (int): Dimension of the input features.
            dim_output (int): Dimension of the output features.
            cfg (Namespace): Configuration containing memory-related parameters.
        """
        super(MemoryRecon, self).__init__()
        self.dim_input = dim_input  # source feature dim
        self.dim_output = dim_output  # target feature dim
        self.cfg = cfg
        # Handle different recon_content scenarios
        if cfg.MODEL.ARCH == "r3d_shallow" or cfg.RESNET.DEPTH <= 34:
            self.recon_dim = cfg.RESNET.WIDTH_PER_GROUP * 8
            self.recon_mid_dim = cfg.RESNET.WIDTH_PER_GROUP * 8
        else:
            self.recon_dim = cfg.RESNET.WIDTH_PER_GROUP * 32
            self.recon_mid_dim = cfg.RESNET.WIDTH_PER_GROUP * 32

        # Initialize various attributes based on configuration
        self._initialize_attributes()

        # Initialize components
        self._initialize_components()

        # Initialize softmax
        self.softmax = nn.Softmax(-1)
        self.pool_feat = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _initialize_attributes(self):
        """
        Initializes attributes based on configuration settings.
        """
        # Extract relevant configuration attributes from memory configuration
        memory_cfg = self.cfg.MEMORY

        # Dimension of the source visual concept dictionary
        self.dim_source_mem = memory_cfg.DIM_SOURCE_MEM
        # Dimension of the target visual concept dictionary
        self.dim_target_mem = memory_cfg.DIM_TARGET_MEM
        # Dimension of the key-value cross memory
        self.dim_cross_mem = memory_cfg.DIM_CROSS_MEM

        # Scaling factor for attention distribution
        self.radius = memory_cfg.RADIUS
        # Number of visual concepts in the dictionaries
        self.n_concept = memory_cfg.N_CONCEPT
        # Number of slots for key and value memories
        self.n_slot = memory_cfg.N_MEMORY
        # Number of attention heads of the memory
        self.n_head = memory_cfg.N_HEAD
        # Type of loss function for feature reconstruction using dictionary learning
        self.recon_loss_type = memory_cfg.RECON_LOSS_TYPE
        # Average dimension of the reconstruction loss
        self.average_dim = memory_cfg.AVERAGE_DIM
        # Whether to use input projection before memory
        self.use_input_proj = memory_cfg.USE_INPUT_PROJ
        # Dimension for input projection to cross memory
        self.dim_cross_input_proj = memory_cfg.DIM_CROSS_INPUT_PROJ
        # Whether to use output projection after memory
        self.use_output_proj = memory_cfg.USE_OUTPUT_PROJ
        # Whether to use source feature reconstruction
        self.use_src_recon = memory_cfg.USE_SRC_RECON
        # Whether to use target feature reconstruction
        self.use_tar_recon = memory_cfg.USE_TAR_RECON
        # Whether to use KL divergence loss for alignment between visual concept codes
        self.use_align = memory_cfg.USE_ALIGN

        # Whether to use contrastive loss for source memory
        self.use_source_mem_contrastive = memory_cfg.USE_SOURCE_MEM_CONTRASTIVE
        # Whether to use contrastive loss for target memory
        self.use_target_mem_contrastive = memory_cfg.USE_TARGET_MEM_CONTRASTIVE
        # Whether to use contrastive loss for cross memory
        self.use_cross_mem_contrastive = memory_cfg.USE_CROSS_MEM_CONTRASTIVE

        # Type of addressing for memory (e.g., "cosine" or other)
        self.address_type = memory_cfg.ADDRESS_TYPE
        # Whether to predict residuals for memory update
        self.predict_residual = memory_cfg.PREDICT_RESIDUAL

        # Whether to use sparse coding for memory
        self.use_sparse = memory_cfg.USE_SPARSE
        # Whether to use sparse coding before memory (not clear)
        self.use_sparse_before = memory_cfg.USE_SPARSE_BEFORE
        # Regulization loss type for sparse coding
        self.sparse_loss_type = memory_cfg.SPARSE_LOSS_TYPE
        # Top-k for sparse coding
        self.sparse_topk = memory_cfg.SPARSE_TOPK

        # Type of the structure for feature reconstruction in the dictionary learning
        self.recon_type = memory_cfg.RECON_TYPE
        # Type of the structure of the predictor, memory for the key-value memory enhanced predictor and mlp for the mlp predictor
        self.cross_branch_type = memory_cfg.CROSS_BRANCH_TYPE
        # Temperature for DINO loss (teacher)
        self.dino_temp_t = memory_cfg.DINO_TEMP_T
        # Temperature for DINO loss (student)
        self.dino_temp_s = memory_cfg.DINO_TEMP_S
        # Number of MLP layers for the dictionary reconstruction branch
        self.num_mlp_layers = memory_cfg.NUM_MLP_LAYERS
        # Number of MLP layers for the source feature projection branch
        self.num_mlp_layers_in = memory_cfg.NUM_MLP_LAYERS_IN
        # Number of MLP layers for the output feature projection branch
        self.num_mlp_layers_out = memory_cfg.NUM_MLP_LAYERS_OUT

    def _initialize_components(self):
        # Initialize source reconstruction components
        if self.use_src_recon:
            self._initialize_source_recon()

        # Initialize target reconstruction components
        if self.use_tar_recon:
            self._initialize_target_recon()

        # Initialize cross-branch components
        if self.cross_branch_type == "memory":
            self._initialize_cross_memory()
        else:
            self._initialize_cross_branch()

        # Initialize input projection components
        if self.use_input_proj:
            self._initialize_input_projection()

        # Initialize output projection components
        if self.use_output_proj:
            self._initialize_output_projection()

        # Initialize KL and loss alignment components
        self._initialize_kl_dino_loss()

    def _initialize_source_recon(self):
        if self.address_type == "cosine":
            self.source_key = nn.utils.weight_norm(
                nn.Linear(self.dim_source_mem, self.n_concept, bias=False))
            self.source_key.weight_g.data.fill_(1)
            self.source_key.weight_g.requires_grad = False
        else:
            self.source_key = nn.Linear(self.dim_source_mem,
                                        self.n_concept,
                                        bias=False)

        if self.recon_type == "memory":
            self.source_mem = nn.Parameter(torch.Tensor(
                self.n_concept, self.dim_source_mem),
                                           requires_grad=True)
            nn.init.trunc_normal_(self.source_mem, std=0.02)
        else:
            self.source_recon = head_helper.MLPHead(
                self.n_concept,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_target_recon(self):
        if self.address_type == "cosine":
            self.target_key = nn.utils.weight_norm(
                nn.Linear(self.dim_target_mem, self.n_concept, bias=False))
            self.target_key.weight_g.data.fill_(1)
            self.target_key.weight_g.requires_grad = False
        else:
            self.target_key = nn.Linear(self.dim_target_mem,
                                        self.n_concept,
                                        bias=False)

        if self.recon_type == "memory":
            self.target_mem = nn.Parameter(torch.Tensor(
                self.n_concept, self.dim_target_mem),
                                           requires_grad=True)
            nn.init.trunc_normal_(self.target_mem, std=0.02)
        else:
            self.target_recon = head_helper.MLPHead(
                self.n_concept,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_cross_memory(self):
        self.cross_key = nn.utils.weight_norm(
            nn.Linear(self.dim_cross_input_proj, self.n_slot, bias=False))
        self.cross_key.weight_g.data.fill_(1)
        self.cross_key.weight_g.requires_grad = False
        self.cross_mem = nn.Parameter(torch.Tensor(self.n_slot,
                                                   self.dim_cross_mem),
                                      requires_grad=True)
        nn.init.trunc_normal_(self.cross_mem, std=0.02)

    def _initialize_cross_branch(self):
        self.cross_branch = head_helper.MLPHead(
            self.dim_input,
            self.dim_output,
            self.cfg.CONTRASTIVE.MLP_DIM,
            2,
            bn_on=self.cfg.CONTRASTIVE.BN_MLP,
            bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
            if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
            global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                         and self.cfg.BN.GLOBAL_SYNC),
        )

    def _initialize_input_projection(self):
        if self.cross_branch_type == "memory":
            self.in_proj_cross = head_helper.MLPHead(
                self.dim_input,
                self.n_head * self.dim_cross_input_proj,
                self.n_head * self.dim_cross_input_proj
                if self.num_mlp_layers_in == 1 else self.dim_cross_input_proj,
                self.num_mlp_layers_in,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

        if self.use_src_recon and self.recon_type == "memory":
            self.in_proj_src = head_helper.MLPHead(
                self.dim_input,
                self.dim_source_mem,
                self.dim_input,
                self.num_mlp_layers_in,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

        if self.use_tar_recon and self.recon_type == "memory":
            self.in_proj_tar = head_helper.MLPHead(
                self.dim_output,
                self.dim_target_mem,
                self.dim_output,
                self.num_mlp_layers_in,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_output_projection(self):
        self.out_proj_cross = head_helper.MLPHead(
            self.n_head * self.dim_cross_mem,
            self.dim_output,
            self.dim_output,
            self.num_mlp_layers_out,
            bn_on=self.cfg.CONTRASTIVE.BN_MLP,
            bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
            if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
            global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                         and self.cfg.BN.GLOBAL_SYNC),
        )

        if self.use_src_recon and self.recon_type == "memory":
            self.out_proj_source = head_helper.MLPHead(
                self.dim_source_mem,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

        if self.use_tar_recon and self.recon_type == "memory":
            self.out_proj_tar = head_helper.MLPHead(
                self.dim_target_mem,
                self.recon_dim,
                self.recon_mid_dim,
                self.num_mlp_layers,
                bn_on=self.cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=self.cfg.BN.NUM_SYNC_DEVICES
                if self.cfg.CONTRASTIVE.BN_SYNC_MLP else 1,
                global_sync=(self.cfg.CONTRASTIVE.BN_SYNC_MLP
                             and self.cfg.BN.GLOBAL_SYNC),
            )

    def _initialize_kl_dino_loss(self):
        self.dino_loss = DINOLoss(
            out_dim=self.n_concept,
            student_temp=self.dino_temp_s,
            teacher_temp=self.dino_temp_t,
            sync_center=True if self.cfg.NUM_GPUS > 1 else False).cuda()

    @staticmethod
    def regularization_loss(self, input, topk=-1):
        try:
            # Try to extract dimensions from the input tensor
            BS, n_head, n_concept = input.shape
        except:
            # If the input has only two dimensions, assume n_head = 1
            BS, n_concept = input.shape
            n_head = 1

        if self.cfg.sparse_loss_type == 'l1':
            # L1-norm regularization loss
            loss = torch.norm(input, p=1) / BS
        elif self.cfg.sparse_loss_type == 'neglog':
            # Negative log-likelihood regularization loss
            loss = torch.mean(-input * torch.log(input))
        elif self.cfg.sparse_loss_type == 'max_margin':
            # Maximum margin regularization loss
            loss = torch.mean(
                torch.max(
                    torch.min(input, dim=0)[0] - torch.max(input, dim=0)[0] +
                    0.1,
                    torch.zeros_like(torch.min(input, dim=0)[0]),
                ))
        elif self.cfg.sparse_loss_type == 'max_margin_neglog':
            # Combination of max margin and negative log-likelihood losses
            topk_ = 10
            loss = 0.01 * torch.sum(
                torch.max(
                    torch.min(input, dim=0)[0] - torch.max(input, dim=0)[0] +
                    0.1,
                    torch.zeros_like(torch.min(input, dim=0)[0]),
                )) + 0.1 * torch.sum(1 - torch.sum(
                    torch.topk(input, topk_, dim=-1)[0], dim=-1)) / (BS *
                                                                     n_head)
        else:
            # Top-k regularization loss
            loss = torch.sum(1 - torch.sum(torch.topk(input, topk, dim=-1)[0],
                                           dim=-1)) / (BS * n_head)
        return loss

    @staticmethod
    def reconstruction_loss(pred, tar, loss_type="cosine", average_dim=0):
        assert pred.size() == tar.size() and tar.numel() > 0
        if loss_type == "l2":
            loss = torch.sum(torch.pow(pred - tar, 2))
        elif loss_type == "cosine":
            loss = torch.abs(1 - F.cosine_similarity(pred, tar, 1)).sum()
        else:
            raise RuntimeError(f"Loss type {loss_type} is not supported.")

        if average_dim == -1:
            loss /= tar.numel()
        else:
            loss /= tar.shape[average_dim]
        return loss

    @staticmethod
    def contrastive_loss(self, input):
        """
        Contrastive loss to encourage distinctiveness of visual features stored in different memory slots.

        Args:
            input (torch.Tensor): A 2D tensor representing the features stored in value memories.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        assert len(input.shape) == 2

        # Compute the confusion matrix as described in the contrastive learning method
        confusion_matrix = torch.abs(
            torch.eye(input.shape[0]).cuda() - torch.matmul(
                F.normalize(input, dim=-1),
                F.normalize(input, dim=-1).transpose(0, 1),
            ))

        # Compute the separate loss by summing up the values in the confusion matrix and dividing by the number of slots
        separate_loss = torch.sum(confusion_matrix) / input.shape[0]

        return separate_loss

    def forward(self, source, infos=None):
        """
        Forward pass of MemoryRecon.

        Args:
            source (Tensor): Input source data.
            infos (list): List of additional information.

        Returns:
            output (dict): Output containing various losses and predictions.
        """

        # Ensure the length of 'infos' is 3
        assert len(infos) == 3

        # Process target data from 'infos'
        if infos[0] is not None:
            if len(infos[0]) > 1:
                target = torch.cat(infos[0]).detach()
            else:
                target = infos[0][0].detach()
        else:
            # For calculating params and flops
            target = source.clone().detach()

        # Initialize variables for features
        feat, feat_target = None, None
        if infos[1] is not None:
            feat = infos[1][0]
        else:
            feat = torch.randn(source.shape[0], self.recon_dim, 1, 1,
                               1).type_as(source)
        if infos[2] is not None:
            if len(infos[2]) > 1:
                feat_target = torch.cat([feat_tar[0]
                                         for feat_tar in infos[2]]).detach()
            else:
                feat_target = infos[2][0][0]
        else:
            feat_target = torch.randn(source.shape[0], self.recon_dim, 1, 1,
                                      1).type_as(source)

        # Get batch size and feature dimension
        B, C = source.size()

        # Initialize output dictionary
        output = {}

        # Calculate projection of source data for cross memory mechanism
        if self.use_input_proj and self.cross_branch_type == "memory":
            source_cross_proj = self.in_proj_cross(source)
        else:
            source_cross_proj = source

        # Calculate cross memory mechanism and address
        if self.cross_branch_type == "memory":
            cross_mem_sim = torch.einsum(
                "bhd,hsd->bhs",
                F.normalize(
                    source_cross_proj.view(B, self.n_head,
                                           self.dim_cross_input_proj),
                    dim=2,
                ),
                F.normalize(
                    self.cross_key.weight_v.view(
                        self.n_head,
                        self.n_slot // self.n_head,
                        self.dim_cross_input_proj,
                    ),
                    dim=2,
                ),
            )
            if self.cfg.TEST.VIS_MIDDLE:
                output["vis_cross_mem_sim"] = cross_mem_sim
            cross_mem_address = self.softmax(self.radius * cross_mem_sim)
            cross_recon = torch.einsum(
                "bhs,hsd->bhd",
                cross_mem_address,
                self.cross_mem.view(
                    self.n_head,
                    self.n_slot // self.n_head,
                    self.dim_cross_mem,
                ),
            ).reshape(B, self.n_head * self.dim_cross_mem)
            if self.predict_residual:
                cross_recon += source
            if self.use_output_proj:
                cross_recon = self.out_proj_cross(cross_recon)
        else:
            cross_recon = self.cross_branch(source)
        output["output_predict"] = cross_recon

        # Perform calculations related to target reconstruction
        if self.use_tar_recon:
            # Check if input projection and reconstruction type are memory-based
            if self.use_input_proj and self.recon_type == "memory":
                target_proj = self.in_proj_tar(target)
            else:
                target_proj = target

            # Calculate target memory similarity using the specified address type
            if self.address_type == "cosine":
                target_mem_sim = self.target_key(
                    F.normalize(target_proj, dim=-1))
            else:
                target_mem_sim = self.target_key(target_proj)

            # Optionally visualize target memory similarity
            if self.cfg.TEST.VIS_MIDDLE:
                output["vis_target_mem_sim"] = target_mem_sim

            # Apply softmax to target memory similarity multiplied by radius
            target_mem_address = self.softmax(self.radius * target_mem_sim)

            # Calculate target reconstruction based on reconstruction type
            if self.recon_type == "memory":
                target_recon = torch.matmul(target_mem_address,
                                            self.target_mem)
                target_recon = self.out_proj_tar(target_recon)
            else:
                target_recon = self.target_recon(target_mem_sim)

            # Calculate the reconstruction loss for target
            loss_target_recon = self.reconstruction_loss(
                target_recon,
                self.pool_feat(feat_target).view(target_recon.shape[0],
                                                 -1).detach(),
                loss_type=self.recon_loss_type,
                average_dim=self.average_dim,
            )
            output["loss_target_recon"] = loss_target_recon

            # Calculate sparse loss for target memory if required
            if self.use_sparse:
                if self.use_sparse_before:
                    # Calculate sparse loss using softmax of target memory similarity
                    loss_target_sparse = self.regularization_loss(
                        self.softmax(target_mem_sim), self.topk)
                else:
                    # Calculate sparse loss using target memory address
                    loss_target_sparse = self.regularization_loss(
                        target_mem_address, self.topk)
                output["loss_target_mem_sparse"] = loss_target_sparse

        # Perform calculations related to source reconstruction
        if self.use_src_recon:
            # Check if input projection and reconstruction type are memory-based
            if self.use_input_proj and self.recon_type == "memory":
                source_proj = self.in_proj_src(source)
            else:
                source_proj = source

            # Calculate source memory similarity using the specified address type
            if self.address_type == "cosine":
                source_mem_sim = self.source_key(
                    F.normalize(source_proj, dim=-1))
            else:
                source_mem_sim = self.source_key(source_proj)

            # Optionally visualize source memory similarity
            if self.cfg.TEST.VIS_MIDDLE:
                output["vis_source_mem_sim"] = source_mem_sim

            # Apply softmax to source memory similarity multiplied by radius
            source_mem_address = self.softmax(self.radius * source_mem_sim)

            # Calculate source reconstruction based on reconstruction type
            if self.recon_type == "memory":
                source_recon = torch.matmul(source_mem_address,
                                            self.source_mem)
                source_recon = self.out_proj_source(source_recon)
            else:
                source_recon = self.source_recon(source_mem_sim)

            # Calculate the reconstruction loss for source
            loss_source_recon = self.reconstruction_loss(
                source_recon,
                self.pool_feat(feat).view(B, -1).detach(),
                loss_type=self.recon_loss_type,
                average_dim=self.average_dim,
            )
            output["loss_source_recon"] = loss_source_recon

            # Calculate sparse loss for source memory if required
            if self.use_sparse:
                if self.use_sparse_before:
                    # Calculate sparse loss using softmax of source memory similarity
                    loss_source_mem_sparse = self.regularization_loss(
                        self.softmax(source_mem_sim), self.topk)
                else:
                    # Calculate sparse loss using source memory address
                    loss_source_mem_sparse = self.regularization_loss(
                        source_mem_address, self.topk)
                output["loss_source_mem_sparse"] = loss_source_mem_sparse

        # Calculate contrastive losses if specified memory types are used
        if self.use_target_mem_contrastive:
            output["loss_target_mem_contrastive"] = self.contrastive_loss(
                self.target_mem)

        if self.use_source_mem_contrastive:
            output["loss_source_mem_contrastive"] = self.contrastive_loss(
                self.source_mem)

        if self.use_cross_mem_contrastive:
            output["loss_cross_mem_contrastive"] = self.contrastive_loss(
                self.cross_mem)

        # Calculate visual concept alignment loss using KL divergence loss
        if self.use_align and self.use_src_recon and self.use_tar_recon:
            loss_kl = self.dino_loss(source_mem_sim, target_mem_sim)
            output["loss_kl"] = loss_kl

        return output
