from utils.data.cornell_data import CornellDataset
from utils.data.grasp_data import GraspDatasetBase
from utils.data.jacquard_data import JacquardDataset


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
        if self.depth_files[idx][-6:] == 'd.tiff':
            return self.cornell_grasp.get_depth(idx)
        else:
            return self.jacquard_dataset.get_depth(idx)

    def get_observation_dataset_name(self, idx):
        if self.depth_files[idx][-6:] == 'd.tiff':
            return 'cornell'
        else:
            return 'jacquard'
