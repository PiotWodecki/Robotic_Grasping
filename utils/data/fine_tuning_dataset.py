import copy
import warnings

import torch

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

    def get_gtbb_by_name(self, name):
        """
        Function to use correct get_gtbb function depending on the dataset
        """
        if name[-8:] == 'cpos.txt':
            return self.cornell_grasp.get_gtbb_by_file_name(name)
        else:
            return self.jacquard_dataset.get_gtbb_by_file_name(name)

    def get_depth(self, idx):
        return self.depth_files[idx]

    def get_gtbb(self, idxs):
        if isinstance(idxs, int):
            return self.grasp_files[idxs]
        else:
            return self.grasp_files[idxs.item()]
        #     # # arr = np.zeros(idxs.shape[0])
        #     arr_for_tensors = []
        #     for i, idx in enumerate(idxs):
        #         arr = self.grasp_files[idx].to_array()
        #         arr_for_tensors.append(arr)
        #     # #     # np.insert(arr, i, self.grasp_files[idx])
        #     # #
        #     arr_for_tensors = np.array(arr_for_tensors)
        #     tensor = torch.tensor(arr_for_tensors)
        #     tensor = tensor.squueze()
            # return np.array(arr, dtype=object)
            # return self.grasp_files[idxs]
        return tensor

    def get_points_of_grasping_rectangle(self):
        pass

    def get_gtbb_for_validation(self, idxs, rot, zoom):
        grasp_rectangles = []
        for idx in idxs:
            if len(self.grasp_files[idx].grs) > 20:
                gtbbs = copy.deepcopy(self.grasp_files[idx])
                for i, gtbb in enumerate(gtbbs):
                    c = self.output_size // 2
                    gtbb.rotate(rot, (c, c))
                    gtbb.zoom(zoom, (c, c))
                    gtbbs[i] = gtbb
                grasp_rectangles.append(gtbbs)
            else:
                gtbbs = copy.deepcopy(self.grasp_files[idx])
                for i, gtbb in enumerate(gtbbs):
                    center, left, top = self._get_crop_attrs(gtbb)
                    gtbb.rotate(rot, center)
                    gtbb.offset((-top, -left))
                    gtbb.zoom(zoom, (self.output_size // 2, self.output_size // 2))
                    gtbbs[i] = gtbb
                grasp_rectangles.append(gtbbs)

        return grasp_rectangles

    def get_observation_dataset_name(self, idx):
        if self.depth_files[idx][-6:] == 'd.tiff':
            return 'cornell'
        else:
            return 'jacquard'

    def get_depth_image_from_name(self, name):
        if name[-6:] == 'd.tiff':
            return self.cornell_grasp.get_depth_by_file_name(name)
        else:
            return self.jacquard_dataset.get_depth_by_file_name(name)

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

    def _get_crop_attrs(self, gtbbs):
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def apply_augmentation_to_dataset_refactored(self, cornell, jacquard):
        fine_tuned_dataset = copy.deepcopy(self)

        for idx in enumerate(self.depth_files):
            indeks = idx[0]
            observation_dataset_name = 'cornell'
            if indeks > len(self.cornell_grasp) -1:
                indeks = indeks - len(self.cornell_grasp)
                observation_dataset_name = 'jacquard'

            rectangles_name = self.get_gtbb(indeks)
            rectangles = self.get_gtbb_by_name(rectangles_name)
            img_name = self.get_depth(indeks)
            img = self.get_depth_image_from_name(img_name)
            rectangles_non_rotated, angles = set_rectangles_angles(rectangles)
            bboxes = rectangles_non_rotated.get_albumentations_coco_bboxes(angles)
            transformed = apply_data_augmentation(image=img, bboxes=bboxes,
                                                  observation_dataset=observation_dataset_name)
            img_transformed, bboxes_transformed = transformed['image'], transformed['bboxes']
            rectangles_rotated = build_rectangle(bboxes_transformed)
            fine_tuned_dataset.grasp_files[idx[0]] = rectangles_rotated
            fine_tuned_dataset.depth_files[idx[0]] = img_transformed

        return fine_tuned_dataset

