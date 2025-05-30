# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    ## For Entity
    cfg.ENTITY = CN()
    cfg.ENTITY.ENABLE = False
    cfg.ENTITY.CROP_AREA_RATIO = 0.7
    cfg.ENTITY.CROP_STRIDE_RATIO = 0.6
    cfg.ENTITY.CROP_SAMPLE_NUM_TRAIN = 1
    cfg.ENTITY.CROP_SAMPLE_NUM_TEST = 4

    ## fuse frame embeddings to batch embedding
    cfg.ENTITY.FUSE_NUM_LAYERS = 1
    cfg.ENTITY.FUSE_ENC_HIDDIEN_DIM = 256
    cfg.ENTITY.FUSE_ENC_NHEADS = 8
    cfg.ENTITY.FUSE_ENC_PRE_NORM = False
    cfg.ENTITY.FUSE_ENC_DIM_FEEDFORWARD = 2048
    cfg.ENTITY.FUSE_ENC_LAST_LAYERS = 1
    cfg.ENTITY.FUSE_DEC_NUM_LAYERS = 3

    ## Hornet backbone
    cfg.MODEL.HORNET = CN()
    cfg.MODEL.HORNET.DEPTHS = [2, 3, 18, 2]
    cfg.MODEL.HORNET.BASE_DIM = 192
    cfg.MODEL.HORNET.GCONV = ['partial(gnconv, order=2, s=1/3)', 'partial(gnconv, order=3, s=1/3)', 'partial(gnconv, order=4, s=1/3, h=24, w=13, gflayer=GlobalLocalFilter)', 'partial(gnconv, order=5, s=1/3, h=12, w=7, gflayer=GlobalLocalFilter)']
    cfg.MODEL.HORNET.DROP_PATH_RATE=0.6
    cfg.MODEL.HORNET.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    ## efficientVMamba backbone
    cfg.MODEL.VSSM = CN()
    cfg.MODEL.VSSM.DROP_PATH_RATE = 0.2
    cfg.MODEL.VSSM.PATCH_SIZE = 4
    cfg.MODEL.VSSM.SHARED_SSM = False
    cfg.MODEL.VSSM.SOFTMAX = False
    cfg.MODEL.VSSM.PATCH_NORM = True
    cfg.MODEL.VSSM.IN_CHANS = 3
    cfg.MODEL.VSSM.EMBED_DIM = 96
    cfg.MODEL.VSSM.DEPTHS = [2, 2, 9, 2]
    cfg.MODEL.VSSM.D_STATE = 16
    cfg.MODEL.VSSM.DT_RANK = "auto"
    cfg.MODEL.VSSM.SSM_RATIO = 2.0
    cfg.MODEL.VSSM.MLP_RATIO = 4.0
    cfg.MODEL.VSSM.DOWNSAMPLE = "v2"
    cfg.MODEL.VSSM.WINDOWSIZE = 2
    cfg.MODEL.VSSM.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    ## OVDIO
    cfg.MODEL.OVDINO = CN()
    cfg.MODEL.OVDINO.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.OVDINO.USE_ENCODER_IDX = [2]
    cfg.MODEL.OVDINO.NUM_ENCODER_LAYER = 1
    cfg.MODEL.OVDINO.IN_CHANNELS = [96, 192, 384]
    cfg.MODEL.OVDINO.FPN_IN_CHANNELS = [96, 192, 384]
    cfg.MODEL.OVDINO.FPN_OUT_CHANNELS = 256
    cfg.MODEL.OVDINO.FPN_EXPANSION = 1.0
    cfg.MODEL.OVDINO.LAYER_ATTEN_EMBED_DIMS = 256
    cfg.MODEL.OVDINO.LAYER_ATTEN_NUM_HEADS = 8
    cfg.MODEL.OVDINO.LAYER_ATTEN_DROPOUT = 0.0
    cfg.MODEL.OVDINO.LAYER_FFN_EMBED_DIMS = 256
    cfg.MODEL.OVDINO.LAYER_FFN_FFD_CHANNELS = 1024
    cfg.MODEL.OVDINO.LAYER_FFN_DROP = 0.0

    ## mobile mamba, b1
    cfg.MODEL.MOMAMBA = CN()
    cfg.MODEL.MOMAMBA.STAGES = ['s', 's', 's', 's']
    cfg.MODEL.MOMAMBA.EMBED_DIM = [200, 376, 448]
    cfg.MODEL.MOMAMBA.GLOBAL_RATIO = [0.8, 0.7, 0.6]
    cfg.MODEL.MOMAMBA.LOCAL_RATIO = [0.2, 0.2, 0.3]
    cfg.MODEL.MOMAMBA.DEPTH = [2, 3, 2]
    cfg.MODEL.MOMAMBA.KERNELS = [7, 5, 3]
    cfg.MODEL.MOMAMBA.DOWN_OPS = [['subsample', 2], ['subsample', 2], ['subsample', 2], ['']]
    cfg.MODEL.MOMAMBA.DROP_PATH = 0.0
    cfg.MODEL.MOMAMBA.SSM_RATIO = 2
    cfg.MODEL.MOMAMBA.OUT_INDICES = (1,2,3)

    ## sparx mamba tiny
    cfg.MODEL.SPARXMAMBA = CN()
    cfg.MODEL.SPARXMAMBA.TYPE = "t"
    cfg.MODEL.SPARXMAMBA.DEPTHS = [2, 2, 7, 2]
    cfg.MODEL.SPARXMAMBA.DIMS = [96, 192, 320, 512]
    cfg.MODEL.SPARXMAMBA.SR_RATIO = [8, 4, 2, 1]
    cfg.MODEL.SPARXMAMBA.STEM_TYPE = "v1"

    ## mlla tiny
    cfg.MODEL.MLLA = CN()
    cfg.MODEL.MLLA.DROP_PATH_RATE = 0.2
    cfg.MODEL.MLLA.EMBED_DIM = 64
    cfg.MODEL.MLLA.DEPTHS = [ 2, 4, 8, 4 ]
    cfg.MODEL.MLLA.NUM_HEADS = [ 2, 4, 8, 16 ]