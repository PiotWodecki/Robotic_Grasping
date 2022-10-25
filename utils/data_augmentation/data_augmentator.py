import albumentations as A
import cv2


def apply_data_augmentation(image, bboxes):
    """
    this function apply data augmewntation to both image and bboxes. Class labels are random values -
    it is required argument and we do not have classes
    :param image: Image
    :param bboxes: Grasp rectangles
    :return: Transformed image and transformed grasp rectangles
    """
    transform = A.Compose([
        # A.ToFloat(max_value=65535.0),
        # A.RandomRotate90(),
        # A.Flip(),
        # A.OneOf([
        #     A.RandomFog(),
        #     A.AdvancedBlur(),
        #     A.Blur(blur_limit=3, p=0.1),
        #     A.MultiplicativeNoise(),
        #     A.MotionBlur(p=0.2),
        #     A.MedianBlur(blur_limit=3, p=0.1)]),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # A.OneOf([
        #     A.OpticalDistortion(p=0.3),
        #     A.GridDistortion(p=0.1),
        # ], p=0.2),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
        #
        # A.FromFloat(max_value=65535.0)

        # A.RandomCrop(width=450, height=450),
        # A.HorizontalFlip(p=0.5),
        # A.Emboss(),
        # A.Affine()
        # A.OneOf([
        #     A.MotionBlur(p=0.8),
        #     A.GaussNoise(p=0.8)]),
        # A.OneOf([
        #     A.VerticalFlip(p=1),
        #     A.HorizontalFlip(p=1),
        #     A.RandomRotate90(p=1)
        # ])
        # A.BBoxSafeRandomCrop(p=0.8),
        # A.RandomBrightnessContrast(p=0.2)
    ], bbox_params=A.BboxParams(format='coco')) # label_fields=['image', 'bboxes']
    transformed = transform(image=image, bboxes=bboxes)
    return transformed