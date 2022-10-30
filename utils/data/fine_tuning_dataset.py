import copy
import warnings

from utils.data.cornell_data import CornellDataset
from utils.data.grasp_data import GraspDatasetBase
from utils.data.jacquard_data import JacquardDataset
# from utils.data.fine_tuning_dataset import FineTuningDataset

import numpy as np

# from utils.data.data_augmentator import apply_data_augmentation
from utils.data_augmentation.data_augmentator import apply_data_augmentation
from utils.data_processing.generate_transformated_bboxes import set_rectangles_angles, build_rectangle
from utils.data_processing.grasp import GraspRectangles

warnings.simplefilter(action='ignore', category=FutureWarning)


class FineTuningDataset(GraspDatasetBase):
    """
    Dataset wrapper for Cornell dataset and sample of Jacquard dataset.
    """
    def __init__(self, cornell_file_path, jacquard_sample_file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(FineTuningDataset, self).__init__(**kwargs)

        self.cornell_grasp = CornellDataset(cornell_file_path)
        self.jacquard_dataset = JacquardDataset(jacquard_sample_file_path)

        self.grasp_files = self.jacquard_dataset.grasp_files + self.cornell_grasp.grasp_files
        self.depth_files = self.cornell_grasp.depth_files + self.jacquard_dataset.depth_files
        self.rgb_files = None

        if self.include_rgb == 1:
            self.rgb_files = self.cornell_grasp.rgb_files + self.jacquard_dataset.rgb_files

        # del self.jacquard_dataset
        # del self.cornell_grasp

    def get_gtbb(self, idx):
        """
        Function to use correct get_gtbb function depending on the dataset
        """
        if self.grasp_files[idx][-8:] == 'cpos.txt':
            return self.cornell_grasp.get_gtbb(idx)
        else:
            return self.jacquard_dataset.get_gtbb(idx)

    def get_depth(self, idx):
        if 'd.tiff' in self.depth_files[idx][-6:]:
            return self.cornell_grasp.get_depth(idx)
        else:
            try:
                return self.jacquard_dataset.get_depth(idx)
            except IndexError:
                idx = idx - len(self.cornell_grasp)
                return self.jacquard_dataset.get_depth(idx)

    def get_observation_dataset_name(self, idx):
        if self.depth_files[idx][-6:] == 'd.tiff':
            return 'cornell'
        else:
            return 'jacquard'

    def get_part_of_dataset(self, start, end):
        splitted_dataset = copy.deepcopy(self)
        l = len(self.grasp_files)
        splitted_dataset.grasp_files = self.grasp_files[int(l * start):int(l * end)]
        splitted_dataset.depth_files = self.depth_files[int(l * start):int(l * end)]
        if self.rgb_files is None:
            pass
        else:
            self.rgb_files = self.rgb_files[int(l * start):int(l * end)]

        return splitted_dataset

    def apply_augmentation_to_dataset(self, train_dataset):
        grasp_rectangles = GraspRectangles()
        grasp_rectangles_transformed = GraspRectangles()
        rectangles_after_augmentation = []
        fine_tuned_dataset = FineTuningDataset()
        for idx, _ in enumerate(train_dataset):
            observation_dataset_name = train_dataset.get_observation_dataset_name(idx)
            rectangles = train_dataset.get_gtbb(idx)
            img = train_dataset.get_depth(idx)
            rectangles_non_rotated, angles = set_rectangles_angles(rectangles)
            bboxes = rectangles_non_rotated.get_albumentations_coco_bboxes(angles)
            # rectangles.show(ax=None, shape=img.shape, img=img)
            # rectangles_non_rotated.show(ax=None, shape=img.shape, img=img)
            transformed = apply_data_augmentation(image=img, bboxes=bboxes,
                                                  observation_dataset=observation_dataset_name)
            img_transformed, bboxes_transformed = transformed['image'], transformed['bboxes']
            rectangles_rotated = build_rectangle(bboxes_transformed)
            # rectangles_rotated.show(ax=None, shape=img_transformed.shape, img=img_transformed)
            fine_tuned_dataset.grasp_files = bboxes_transformed
            fine_tuned_dataset.depth_files = img_transformed

    def apply_augmentation_to_dataset_refactored(self, cornell, jacquard):
        fine_tuned_dataset = FineTuningDataset(cornell, jacquard)
        for idx in enumerate(self.depth_files):
            indeks = idx[0]
            observation_dataset_name = 'cornell'
            if indeks > len(self.cornell_grasp) -1:
                indeks = indeks - len(self.cornell_grasp)
                observation_dataset_name = 'jacquard'

            print(indeks)
            rectangles = self.get_gtbb(indeks)
            img = self.get_depth(indeks)
            rectangles_non_rotated, angles = set_rectangles_angles(rectangles)
            bboxes = rectangles_non_rotated.get_albumentations_coco_bboxes(angles)
            transformed = apply_data_augmentation(image=img, bboxes=bboxes,
                                                  observation_dataset=observation_dataset_name)
            img_transformed, bboxes_transformed = transformed['image'], transformed['bboxes']
            fine_tuned_dataset.grasp_files[idx[0]] = bboxes_transformed
            fine_tuned_dataset.depth_files[idx[0]] = img_transformed

            #reset indeksu??????? co jak wchodzi do jacquarda i jest out of range

        return fine_tuned_dataset

