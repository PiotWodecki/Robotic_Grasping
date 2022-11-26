import albumentations as A


def apply_data_augmentation(image, bboxes, print_augmentation=False):
    """
    this function apply data augmentation to both image and bboxes. Cornell dataset always has max 10 rectangles,
    flips and rotates dont work for jacquards observation - that is why there is implemented two different transofrmations
    :param image: Image
    :param bboxes: Grasp rectangles
    :return: Transformed image and transformed grasp rectangles
    """
    if len(bboxes) > 20:
        transform = A.ReplayCompose([
            A.OneOf([
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ], p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.AdvancedBlur(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.MultiplicativeNoise(p=0.2),
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2)], p=0.5),
            A.PixelDropout(p=0.3)
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.ReplayCompose([
            A.ElasticTransform(p=0.2),
            A.Perspective(p=0.2),
            A.RandomBrightnessContrast(p=0.1),
            A.OneOf([
                A.AdvancedBlur(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.MultiplicativeNoise(p=0.2),
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2)], p=0.3),
            A.PixelDropout(p=0.3, dropout_prob=0.03)
        ], bbox_params=A.BboxParams(format='coco'))

    transformed = transform(image=image, bboxes=bboxes)

    if print_augmentation:
        print(print_applied_augmentations(transformed))

    return transformed


def print_applied_augmentations(transformed_image):
    return transformed_image['replay']
