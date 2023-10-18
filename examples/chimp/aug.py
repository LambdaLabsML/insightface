import albumentations as A
import cv2


def hue_jitter(hue):
    list_transform = []
    for h in hue:
        transform = A.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=0,
            hue=(h, h),
            p=1.0
        )
        list_transform.append(transform)
    return list_transform


def gaussian_blur(sigma):
    list_transform = []
    for s in sigma:
        transform = A.GaussianBlur(
            p=1.0,
            blur_limit=0,
            sigma_limit=(s, s),
        )
        list_transform.append(transform)
    return list_transform


def compression(quality):
    list_transform = []
    for q in quality:
        transform = A.ImageCompression(
            p=1.0,
            quality_lower=q,
            quality_upper=q,
            )
        list_transform.append(transform)
    return list_transform


def h_shift(shift_x):
    list_transform = []
    for x in shift_x:
        transform = A.ShiftScaleRotate(
            p=1.0,
            shift_limit_x=(x, x),
            shift_limit_y=(0, 0),
            scale_limit=(0, 0),
            rotate_limit=(0, 0),
            border_mode=cv2.BORDER_REPLICATE,
            )
        list_transform.append(transform)
    return list_transform


def v_shift(shift_y):
    list_transform = []
    for y in shift_y:
        transform = A.ShiftScaleRotate(
            p=1.0,
            shift_limit_x=(0, 0),
            shift_limit_y=(y, y),
            scale_limit=(0, 0),
            rotate_limit=(0, 0),
            border_mode=cv2.BORDER_REPLICATE,
            )
        list_transform.append(transform)
    return list_transform


def roll(rotate_roll):
    list_transform = []
    for r in rotate_roll:
        transform = A.ShiftScaleRotate(
            p=1.0,
            shift_limit_x=(0, 0),
            shift_limit_y=(0, 0),
            scale_limit=(0, 0),
            rotate_limit=(r, r),
            border_mode=cv2.BORDER_REPLICATE,
            )
        list_transform.append(transform)
    return list_transform    
