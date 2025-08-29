import albumentations as A
try:
    from albumentations.augmentations.transforms import ImageCompression
except ImportError:
    # Fallback: define a dummy ImageCompression if albumentations is not installed
    class ImageCompression:
        def __init__(self, quality_lower=30, quality_upper=95, p=0.5):
            pass
        def __call__(self, *args, **kwargs):
            return args[0] if args else None
from albumentations.pytorch import ToTensorV2

def bleed_through(image, alpha=0.3):
    # placeholder for bleed-through custom op: in real code you'd composite mirrored text masks
    return image

def doc_train_augs(strong: bool = True):
def get_doc_augmentations(strong: bool = True):
    prob = 0.7 if strong else 0.3
    return A.Compose([
        A.Perspective(scale=(0.02, 0.08), p=prob*0.6),
        A.Rotate(limit=15, p=prob),
        A.RandomBrightnessContrast(p=prob),
        A.GaussianBlur(blur_limit=(1,5), p=prob*0.4),
        A.GaussNoise(var_limit=(5.0, 50.0), p=prob*0.5),
        ImageCompression(quality_lower=30, quality_upper=95, p=prob),
        A.Downscale(scale_min=0.6, scale_max=0.95, p=prob*0.3),
        A.MotionBlur(p=prob*0.2),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=prob*0.2),
        A.CLAHE(p=prob*0.2),
        A.ToGray(p=0.1),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def apply_augmentations(image, bboxes=None, category_ids=None, strong=True):
    aug = get_doc_augmentations(strong)
    bboxes = bboxes if bboxes is not None else []
    category_ids = category_ids if category_ids is not None else []
    return aug(image=image, bboxes=bboxes, category_ids=category_ids)

if __name__ == '__main__':
    import cv2
    import numpy as np
    img = 255 * np.ones((800, 600, 3), dtype='uint8')
    aug = doc_train_augs(True)
    res = aug(image=img, bboxes=[], category_ids=[])
    print('Augmented image shape:', res['image'].shape)
