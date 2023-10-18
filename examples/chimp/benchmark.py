import argparse
import glob
import os
from pathlib import Path
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import pickle
from trainer_synthetics import FaceSynthetics
from datetime import datetime
import torch

from aug import h_shift, v_shift, roll, compression, gaussian_blur, hue_jitter


from scrfd import SCRFD


num_sample = 10
shift_limit_x = 0.05
shift_limit_y = 0.05
roll_limit = 30
compression_quality_lower = 10
gaussian_sigma_upper = 10.0
hue_limit = 0.5
input_size = 256

shift_x = np.linspace(start=-shift_limit_x, stop=shift_limit_x, num=num_sample)
shift_y = np.linspace(start=-shift_limit_y, stop=shift_limit_y, num=num_sample)
rotate_roll = np.linspace(start=-roll_limit, stop=roll_limit, num=num_sample)
compression_quality = np.linspace(start=compression_quality_lower, stop=100, num=num_sample).astype(int)
gaussian_sigma = np.linspace(start=0, stop=gaussian_sigma_upper, num=num_sample)
hue = np.linspace(start=-hue_limit, stop=hue_limit, num=num_sample)

def visualize(image, trans, flag_save=False, output_dir=None):
    transformed_image = trans(image=image)["image"]
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(transformed_image)
      
    if flag_save:
        name = datetime.now().isoformat(timespec='microseconds') 
        plt.savefig(output_dir + "/imgs_aug" + '/' + name + '.png', bbox_inches='tight')
    else:
        plt.show()


def detect(image, trans, model):

    # expect a square image
    ori_size = image.shape[0]

    pts2d = np.zeros((2, 68))

    # face detection and landmark detection
    transformed_image = trans(image=image)["image"]
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    transformed_image = cv2.resize(
        transformed_image, (input_size, input_size), interpolation = cv2.INTER_LINEAR)

    input = np.transpose(transformed_image, (2, 0, 1))
    input = np.expand_dims(input, 0)
    imgs = torch.Tensor(input).cuda()
    imgs.div_(255).sub_(0.5).div_(0.5)
    pred = model(imgs).detach().cpu().numpy().flatten().reshape( (-1, 2) )
    pred[:, 0:2] += 1
    pred[:, 0:2] *= (input_size // 2)
    pred = pred.astype(np.int)
    pts2d = np.transpose(pred, (1, 0))
    pts2d = pts2d * ori_size / input_size

    # inverse ShiftScaleRotate if needed
    # Skip the case where no face has been detected. So they won't be used in later statistics analysis
    if isinstance(trans, A.ShiftScaleRotate) and (np.sum(pts2d) != 0):
        shift_x = trans.shift_limit_x[0]
        shift_y = trans.shift_limit_y[0]
        scale = trans.scale_limit[0] - 1.0
        rotate = trans.rotate_limit[0]
        M = cv2.getRotationMatrix2D(center = (image.shape[0]/2, image.shape[1]/2),
                                    angle = -rotate,
                                    scale = 1.0 - scale)
        M[0][2] = M[0][2] - shift_x * image.shape[1]
        M[1][2] = M[1][2] - shift_y * image.shape[0]

        pts3d = np.concatenate((pts2d, np.ones((1, pts2d.shape[1]))), axis=0)
        pts2d = np.matmul(M, pts3d)

    
    return pts2d


def apply(image, transform, model=None, flag_detect=False, flag_save=False, output_dir=None):

    if flag_detect:
        list_pts2d = []
        for trans in transform:
            pts2d = detect(image, trans, model)
            list_pts2d.append(pts2d)
        
        pts2d = np.stack(list_pts2d, axis=0)

        return pts2d
    else:
        for trans in transform:
            visualize(image, trans, flag_save, output_dir=output_dir)


def main(args):

    list_transform_h_shift = h_shift(shift_x)
    list_transform_v_shift = v_shift(shift_y)
    list_transform_roll = roll(rotate_roll)
    list_transform_compression = compression(compression_quality)
    list_transform_gaussian_blur = gaussian_blur(gaussian_sigma)
    list_transform_hue = hue_jitter(hue)

    list_transform = {
        'h_shift': list_transform_h_shift,
        'v_shift': list_transform_v_shift,
        'roll': list_transform_roll,
        'compression': list_transform_compression,
        'gaussian': list_transform_gaussian_blur,
        'hue_jitter': list_transform_hue
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir + "/imgs_aug").mkdir(parents=True, exist_ok=True)
    if args.flag_detect:
        model = FaceSynthetics.load_from_checkpoint(args.model_path).cuda()
        model.eval()

        list_image = {}
        for f in glob.glob(os.path.join(args.input_dir, "*.") + args.format):
            print(f)

            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            list_results = {}
            for transform in list_transform:
                list_results[transform] = apply(
                    image,
                    list_transform[transform],
                    model,
                    flag_detect=True
                )
            list_image[f] = list_results

        os.makedirs(args.output_dir, exist_ok=True)

        with open(os.path.join(args.output_dir, 'list_transform_' + args.model_name + '.pickle'), 'wb') as handle:
            pickle.dump(list_transform, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(args.output_dir, 'list_image_' + args.model_name + '.pickle'), 'wb') as handle:
            pickle.dump(list_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        for f in glob.glob(os.path.join(args.input_dir, "*.") + args.format):
            print(f)

            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            for transform in list_transform:
                apply(
                    image,
                    list_transform[transform],
                    args.flag_detect,
                    flag_save=args.flag_save,
                    output_dir=args.output_dir
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        help="Input folder",
        dest="input_dir",
        type=str
    )

    parser.add_argument(
        "--output-dir",
        help="Output folder",
        dest="output_dir",
        type=str
    )

    parser.add_argument(
        "--model-name",
        help="Name of detection model",
        dest="model_name",
        type=str
    )

    parser.add_argument(
        "--model-path",
        help="Name of detection model",
        dest="model_path",
        type=str
    )

    parser.add_argument(
        "--format",
        help="Input image format",
        choices=["png", "jpg"],
        default="png"
    )

    parser.add_argument(
        "--flag-detect",
        action="store_true",
        dest="flag_detect",
        help="Flag to run detection. If false then simply render the augmented images"
    )

    parser.add_argument(
        "--flag-save",
        action="store_true",
        dest="flag_save",
        help="Flag to save images in visualization mode, instead of display it"
    )

    args = parser.parse_args()

    print(args)
    main(args)