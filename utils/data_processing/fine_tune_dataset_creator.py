import numpy as np

from utils.data_augmentation.data_augmentator import apply_data_augmentation
from utils.data_processing.generate_transformated_bboxes import set_rectangles_angles, build_rectangle
from utils.data_processing.grasp import GraspRectangles


def apply_augmentation_to_dataset(train_dataset):
    grasp_rectangles = GraspRectangles()
    grasp_rectangles_transformed = GraspRectangles()
    rectangles_after_augmentation = []
    for idx, _ in enumerate(train_dataset):
        observation_dataset_name =train_dataset.get_observation_dataset_name(idx)
        rectangles = train_dataset.get_gtbb(idx)
        img = train_dataset.get_depth(idx)
        rectangles_non_rotated, angles = set_rectangles_angles(train_dataset.get_gtbb(idx))
        bboxes = rectangles_non_rotated.get_albumentations_coco_bboxes(angles)
        rectangles.show(ax=None, shape=img.shape, img=img)
        rectangles_non_rotated.show(ax=None, shape=img.shape, img=img)
        transformed = apply_data_augmentation(image=img, bboxes=bboxes, observation_dataset=observation_dataset_name)
        img_transformed, bboxes_transformed = transformed['image'], transformed['bboxes']
        rectangles_rotated = build_rectangle(bboxes_transformed)
        rectangles_rotated.show(ax=None, shape=img_transformed.shape, img=img_transformed)
