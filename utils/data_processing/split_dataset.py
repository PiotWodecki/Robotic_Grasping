import torch

from utils.data import get_dataset
from utils.data.fine_tuning_dataset import FineTuningDataset
from utils.data.cornell_data import CornellDataset
from utils.data.jacquard_data import JacquardDataset


def build_regular_training_dataset(args):
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)

    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_data, val_data


def build_fine_tuning_dataset(args):
    Dataset = get_dataset(args.fine_tuning_dataset_name)
    if type(Dataset) is type(JacquardDataset):
        train_dataset = FineTuningDataset(cornell_file_path=args.dataset_path, jacquard_sample_file_path=args.fine_tuning_dataset_path, ds_rotate=args.ds_rotate,
                                          random_rotate=True, random_zoom=True,
                                          include_depth=args.use_depth, include_rgb=args.use_rgb,
                                          start=0, end=args.split)

        train_dataset.shuffle_dataset()

        train_dataset = train_dataset.apply_augmentation_to_dataset_refactored(args.dataset_path,
                                                                               args.fine_tuning_dataset_path)

        train_data = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_dataset = FineTuningDataset(cornell_file_path=args.dataset_path, jacquard_sample_file_path=args.fine_tuning_dataset_path, ds_rotate=args.ds_rotate,
                                        random_rotate=True, random_zoom=True,
                                        include_depth=args.use_depth, include_rgb=args.use_rgb,
                                        start=args.split, end=1)
        val_dataset.shuffle_dataset()

        val_dataset = val_dataset.fix_validation_without_augmentation()

        val_data = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers
        )

        return train_data, val_data

    elif type(Dataset) is type(CornellDataset):
        train_dataset = FineTuningDataset(jacquard_sample_file_path=args.dataset_path,
                                          cornell_file_path=args.fine_tuning_dataset_path,
                                          ds_rotate=args.ds_rotate,
                                          random_rotate=True, random_zoom=True,
                                          include_depth=args.use_depth, include_rgb=args.use_rgb,
                                          start=0, end=args.split)

        train_dataset.shuffle_dataset()

        train_dataset = train_dataset.apply_augmentation_to_dataset_refactored(args.dataset_path,
                                                                               args.fine_tuning_dataset_path)

        train_data = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_dataset = FineTuningDataset(jacquard_sample_file_path=args.dataset_path,
                                        cornell_file_path=args.fine_tuning_dataset_path, ds_rotate=args.ds_rotate,
                                        random_rotate=True, random_zoom=True,
                                        include_depth=args.use_depth, include_rgb=args.use_rgb,
                                        start=args.split, end=1)
        val_dataset.shuffle_dataset()

        val_dataset = val_dataset.fix_validation_without_augmentation()

        val_data = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers
        )

        return train_data, val_data
    else:
        raise NotImplementedError('Dataset Type is Not implemented')


def build_transfer_learning_dataset(args):
    Dataset = get_dataset(args.transfer_learning_dataset_name)
    train_dataset = Dataset(args.transfer_learning_dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.transfer_learning_dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_data, val_data


def build_dataset(args):
    if args.fine_tuning_dataset_path is None and args.transfer_learning_dataset_path is None:
        return build_regular_training_dataset(args)

    if args.fine_tuning_dataset_path:
        return build_fine_tuning_dataset(args)

    if args.transfer_learning_dataset_path:
        return build_transfer_learning_dataset(args)

    return build_regular_training_dataset(args)
