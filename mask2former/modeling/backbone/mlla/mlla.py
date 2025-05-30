# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/backbones/swin_transformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .backbone_mlla import MLLABackbone


@BACKBONE_REGISTRY.register()
class MLLA(MLLABackbone, Backbone):
    def __init__(self, config, input_shape):
        super().__init__(embed_dim=config.MODEL.MLLA.EMBED_DIM,
                         depths=config.MODEL.MLLA.DEPTHS,
                         num_heads=config.MODEL.MLLA.NUM_HEADS,
                         drop_path_rate=config.MODEL.MLLA.DROP_PATH_RATE,
                         img_size=config.INPUT.IMAGE_SIZE
                        )

        self._out_features = ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32
        }

        self._out_feature_channels = {
            "res2": 64,
            "res3": 128,
            "res4": 256,
            "res5": 512,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        # return y
        for i, k in enumerate(self._out_features):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

    @property
    def size_divisibility(self):
        return 32

