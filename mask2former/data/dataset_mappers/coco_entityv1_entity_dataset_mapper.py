# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch.nn.functional as F
from typing import List, Optional, Union
import torch
import os
import cv2

from detectron2.config import configurable 
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
from pycocotools import mask as coco_mask

from panopticapi.utils import rgb2id, id2rgb

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["Entityv1CocoDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE
    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                         vertical=cfg.INPUT.RANDOM_FLIP == "vertical",)
        )

    augmentation.extend([
        T.ResizeScale(min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints
    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m], m


def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        mask = annotation["segmentation"]
        mask = transforms.apply_segmentation(mask)
        assert tuple(mask.shape[:2]) == image_size
        annotation["segmentation"] = mask

    return annotation


class Entityv1CocoDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train,
        *,
        tfm_gens,
        image_format,
        mask_format
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        # fmt: off
        self.is_train               = is_train
        self.tfm_gens               = tfm_gens
        self.image_format           = image_format
        self.mask_format            = mask_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(tfm_gens))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        tfm_gens = build_transform_gen(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            tfm_gens.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "mask_format": cfg.INPUT.MASK_FORMAT,
        }
        return ret 

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict
        
        # load instance mask
        name  = dataset_dict["file_name"].split("/")[-1].split(".")[0]
        
        #
        if self.is_train:
            panoptic_annotation_path = os.path.join("/xxxx/entity_train2017", name + ".npz")
        else:
            panoptic_annotation_path = os.path.join("/xxxx/entity_val2017", name + ".npz")
        panoptic_semantic_map = np.load(panoptic_annotation_path)
        # x1,y1,x2,y2,category,thing_or_stuff,instance_id
        bounding_boxes = panoptic_semantic_map["bounding_box"].astype(np.float32)
        
        info_map       = panoptic_semantic_map["map"]
        instance_map   = info_map[0]
        semantic_map   = info_map[1]
        num_instances  = len(dataset_dict["annotations"])

        annos = []
        dataset_dict.pop("annotations")
        for i, bbox_info in enumerate(bounding_boxes):
            x1, y1, x2, y2, category, thing, instance_id = bbox_info
            entity_mask = (instance_map == int(instance_id)).astype(np.uint8)
            x1, y1, x2, y2 = int(x1), int(x2), int(y1), int(y2)
            w  = x2 - x1
            h  = y2 - y1
            
            bbox_dict = {"iscrowd": 0,
                         "bbox": [x1, y1, w, h],
                         "category_id": int(category),
                         "bbox_mode": BoxMode.XYWH_ABS,
                         "segmentation": entity_mask
                         }
            transed_bbox_info = transform_instance_annotations(bbox_dict, transforms, image_shape)
            annos.append(transed_bbox_info)

        # NOTE: does not support BitMask due to augmentation
        # Current BitMask cannot handle empty objects
        instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)
        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        # Need to filter empty instances first (due to augmentation)
        instances = utils.filter_empty_instances(instances)
        # Generate masks from polygon
        h, w = instances.image_size
        # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
        if hasattr(instances, 'gt_masks'):
            gt_masks = instances.gt_masks
            if self.mask_format == "polygons":
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            else:
                gt_masks = gt_masks.tensor
            instances.gt_masks = gt_masks
        dataset_dict["instances"] = instances
        return dataset_dict

        aug_input  = ItemAugInput(image, seg_info=seg_info)
        transforms = aug_input.apply_augmentations(self.augmentations)

        image        = aug_input.image
        instance_map = aug_input.seg_info["instance_map"].copy()
        semantic_map = aug_input.seg_info["semantic_map"]

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        new_anns = dataset_dict.pop("annotations")
        new_anns = [obj for obj in new_anns if obj.get("iscrowd", 0) == 0]
        # assert len(new_anns) == bounding_boxes.shape[0], print("{}:{}".format(len(new_anns), bounding_boxes.shape[0]))
        
        for i in range(len(new_anns)):
            x1, y1, x2, y2, category, thing, instance_id = bounding_boxes[i]
            entity_mask = (instance_map == int(i)).astype(np.uint8)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            w  = x2 - x1
            h  = y2 - y1
            new_anns[i]["bbox"] = [x1, y1, w, h]
            new_anns[i]["category_id"] = int(category)


        isthing_list = []
        instance_id_list = []
        for i in range(len(new_anns)):
            x1, y1, x2, y2, category, thing, instance_id = bounding_boxes[i]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            w  = x2 - x1
            h  = y2 - y1
            new_anns[i]["bbox"] = [x1, y1, w, h]
            new_anns[i]["category_id"] = int(category)
            isthing_list.append(int(thing))
            instance_id_list.append(int(instance_id))

        isthing_list = torch.tensor(isthing_list, dtype=torch.int64)
        instance_id_list = torch.tensor(instance_id_list, dtype=torch.int64)

        annos = [utils.transform_instance_annotations(obj, transforms, image_shape) for obj in new_anns if obj.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        instances.instanceid = instance_id_list

        instances, select = filter_empty_instances(instances)

        dataset_dict["instances"] = instances
        dataset_dict["instance_map"] = torch.as_tensor(np.ascontiguousarray(instance_map))
        return dataset_dict
