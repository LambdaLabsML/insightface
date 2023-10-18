from __future__ import division
import numpy as np
import os
import cv2
import glob
import torch
import argparse
import glob
import datetime
import pickle
import shutil

from trainer_synthetics import FaceSynthetics
from insightface.utils import face_align
from insightface.app import FaceAnalysis
from scrfd import SCRFD

# lmrk detector
input_size = 256
USE_FLIP = False

# dataset
dataset_output_size = 384

flip_parts = (
    [1, 17],
    [2, 16],
    [3, 15],
    [4, 14],
    [5, 13],
    [6, 12],
    [7, 11],
    [8, 10],
    [18, 27],
    [19, 26],
    [20, 25],
    [21, 24],
    [22, 23],
    [32, 36],
    [33, 35],
    [37, 46],
    [38, 45],
    [39, 44],
    [40, 43],
    [41, 48],
    [42, 47],
    [49, 55],
    [50, 54],
    [51, 53],
    [62, 64],
    [61, 65],
    [68, 66],
    [59, 57],
    [60, 56],
)

# app = FaceAnalysis()
# app.prepare(ctx_id=0, det_size=(224, 224))

def copy_detection_metadata(dataset_dir, output_dir):
    """
    Compare the image filenames in dataset_dir and output_dir.
    If a match is found, copy bounding box and landmark metadata from output_dir to dataset_dir.
    """
    img_paths = glob.glob(os.path.join(dataset_dir, "*.png"))
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        source_filename = os.path.join(output_dir, filename)
        if os.path.exists(source_filename):
            bbox_path = source_filename.replace(".png", "_bbox.txt")
            ldmks_path = source_filename.replace(".png", "_ldmks.txt")
            shutil.copy(bbox_path, dataset_dir)
            shutil.copy(ldmks_path, dataset_dir)
            print("cp meta files:", filename)


def process_images(dataset_dir, output_dir, dataset_output_size=dataset_output_size, bbox_size_scale=1.5):
    """
    Process images: align the faces, save the images and their landmarks.
    """
    img_paths = glob.glob(os.path.join(dataset_dir, "*.png"))
    X, Y = [], []

    for img_path in img_paths:
        filename = img_path.split("/")[-1]
        img_path = output_dir + "/" + filename
        img = cv2.imread(img_path)
        landmarks = load_landmarks(img_path)
        bbox = load_bounding_box(img_path)
        aimg, pred = align_face_and_landmarks(img, bbox, landmarks, dataset_output_size, bbox_size_scale)
        
        x = os.path.basename(img_path).replace("png", "jpg")
        X.append(x)
        Y.append([(point[0], point[1]) for point in pred])
        cv2.imwrite(os.path.join(dataset_dir, x), aimg)

    with open(os.path.join(dataset_dir, "annot.pkl"), "wb") as pfile:
        pickle.dump((X, Y), pfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_landmarks(img_path):
    """
    Load facial landmarks from the corresponding file.
    """
    ylines = open(img_path.replace(".png", "_ldmks.txt")).readlines()
    ylines = ylines[:68]
    return [tuple(map(float, yline.strip().split())) for yline in ylines]


def load_bounding_box(img_path):
    """
    Load bounding box from the corresponding file.
    """
    bbox_path = img_path.replace(".png", "_bbox.txt")
    with open(bbox_path, "r") as f:
        bbox = list(map(int, f.readline().rstrip("\n").split()))
    return np.array(bbox)


def align_face_and_landmarks(img, bbox, landmarks, output_size, bbox_size_scale):
    """
    Align face in the image based on the bounding box and landmarks.
    """
    dimg = img.copy()
    pred = np.array(landmarks)

    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    rotate = 0
    _scale = output_size / (max(w, h) * bbox_size_scale)
    aimg, M = face_align.transform(dimg, center, output_size, _scale, rotate)
    pred = face_align.trans_points(pred, M)

    return aimg, pred





if __name__ == "__main__":
    # take a folder of images (frames) as input
    # output a folder of images with bbox and landmark annotations
    # the output folder will be created by the bbox detector, with the cropped face images and _bbox.txt
    # each detected face will be padded to a square and resized to 512x512
    # the landmark detector will run landmark detection for each face, and save the result in _ldmks.txt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image_dir",
        type=str,
        default="/home/ubuntu/Chimp/datasets/Chimp_40",
    )
    parser.add_argument(
        "--bbox_detector_path",
        type=str,
        default="/home/ubuntu/Chimp/models/scrfd/scrfd_10g.onnx",
    )
    parser.add_argument(
        "--ldmks_detector_path",
        type=str,
        default="/home/ubuntu/Chimp/models/synthetic_resnet50d.ckpt",
    )
    parser.add_argument("--bbox_confidence", type=float, default=0.3)
    parser.add_argument("--bbox_size_scale", type=float, default=1.5)
    parser.add_argument("--stage", choices=["bbox", "ldmks", "dataset"], default="bbox")
    parser.add_argument("--bbox-method", choices=["app", "scrfd"], default="scrfd")
    parser.add_argument("--dataset_postfix", type=str, default="_dataset")
    args = parser.parse_args()

    detector_name = args.bbox_detector_path.split("/")[-1][:-5]

    output_dir = (
        args.input_image_dir
        + "_"
        + detector_name
        + "_"
        + str(args.bbox_confidence)
        + "_"
        + str(args.bbox_size_scale)
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    render_dir = output_dir + "_render"
    os.makedirs(
        render_dir,
        exist_ok=True,
    )

    dataset_dir = output_dir + args.dataset_postfix
    os.makedirs(
        dataset_dir,
        exist_ok=True,
    )

    if args.stage == "bbox":
        detector = SCRFD(model_file=args.bbox_detector_path)
        detector.prepare(-1)

        # Use glob to find all .jpg and .png files in the specified directory
        img_paths = glob.glob(os.path.join(args.input_image_dir, "*.jpg")) + glob.glob(
            os.path.join(args.input_image_dir, "*.png")
        )

        # bbox detection
        for img_path in img_paths:
            img = cv2.imread(img_path)

            if args.bbox_method == "app":
                dimg = img.copy()
                faces = app.get(img, max_num=1)

                print(img_path)
                if len(faces) != 1:
                    continue
                bbox = faces[0].bbox
                x1, y1, x2, y2 = bbox.astype(int)

                w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
                center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
                rotate = 0
                _scale = input_size / (max(w, h) * args.bbox_size_scale)
                aimg, M = face_align.transform(img, center, input_size, _scale, rotate)

                filename = img_path.split("/")[-1]
                filename = ".".join(filename.split(".")[:-1]) + "_%d.png" % 0
                print("output:", filename)
                cv2.imwrite(output_dir + "/%s" % filename, aimg)

                # write bbox to file
                # for app detector, the bbox is 0, 0, 255, 255
                with open(
                    output_dir + "/%s_bbox.txt" % filename[:-4],
                    "w",
                ) as f:
                    f.write(
                        "%d %d %d %d\n"
                        % (
                            0,
                            0,
                            input_size - 1,
                            input_size - 1,
                        )
                    )

            else:
                for _ in range(1):
                    ta = datetime.datetime.now()
                    bboxes, kpss = detector.detect(
                        img, args.bbox_confidence, input_size=(640, 640)
                    )
                    tb = datetime.datetime.now()
                    print("all cost:", (tb - ta).total_seconds() * 1000)
                print(img_path, bboxes.shape)
                if kpss is not None:
                    print(kpss.shape)
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i]
                    for j in range(len(bbox)):
                        if bbox[j] < 0:
                            bbox[j] = 0

                    x1, y1, x2, y2, score = bbox.astype(int)

                    # compute the center of the bounding box
                    center_x = int((x1 + x2) / 2.0)
                    center_y = int((y1 + y2) / 2.0)

                    # cmpute the width and height of the bounding box
                    width = int(x2 - x1)
                    height = int(y2 - y1)

                    # find the larger side of the bounding box
                    larger_side = int(max(width, height) * args.bbox_size_scale)

                    # get the x1, y1, x2, y2 of the new bounding box, which is a square with the same center
                    x1 = center_x - int(larger_side / 2.0)
                    y1 = center_y - int(larger_side / 2.0)
                    x2 = center_x + int(larger_side / 2.0)
                    y2 = center_y + int(larger_side / 2.0)

                    # crop the image to the new bounding box, if the new bounding box is out of the image, then pad the image with 0
                    x_offset = 0
                    y_offset = 0

                    if x1 < 0:
                        x_offset = -x1
                        x1 = 0
                    if y1 < 0:
                        y_offset = -y1
                        y1 = 0
                    if x2 > img.shape[1]:
                        x2 = img.shape[1]
                    if y2 > img.shape[0]:
                        y2 = img.shape[0]

                    roi = img[y1:y2, x1:x2, :]

                    result_image = np.zeros(
                        (larger_side, larger_side, 3), dtype=np.uint8
                    )

                    # Paste the cropped region into the center of the new image
                    result_image[
                        y_offset : y_offset + roi.shape[0],
                        x_offset : x_offset + roi.shape[1],
                    ] = roi

                    desired_size = 512
                    result_image = cv2.resize(
                        result_image, (desired_size, desired_size)
                    )

                    filename = img_path.split("/")[-1]
                    filename = ".".join(filename.split(".")[:-1]) + "_%d.png" % i
                    print("output:", filename)
                    cv2.imwrite(output_dir + "/%s" % filename, result_image)

                    x1 = larger_side / 2.0 - width / 2.0
                    y1 = larger_side / 2.0 - height / 2.0
                    x2 = larger_side / 2.0 + width / 2.0
                    y2 = larger_side / 2.0 + height / 2.0

                    x1 = x1 * desired_size / larger_side
                    y1 = y1 * desired_size / larger_side
                    x2 = x2 * desired_size / larger_side
                    y2 = y2 * desired_size / larger_side

                    # write bbox to file x1, y1, x2, y2 in the padded image
                    with open(
                        output_dir + "/%s_bbox.txt" % filename[:-4],
                        "w",
                    ) as f:
                        f.write(
                            "%d %d %d %d\n"
                            % (
                                x1,
                                y1,
                                x2,
                                y2,
                            )
                        )

    if args.stage == "ldmks":
        # landmark detection
        ldmks_detector = FaceSynthetics.load_from_checkpoint(
            args.ldmks_detector_path
        ).cuda()
        ldmks_detector.eval()
        # Use glob to find all .jpg and .png files in the specified directory
        img_paths = glob.glob(os.path.join(output_dir, "*.png"))

        for img_path in img_paths:
            print(img_path)
            img = cv2.imread(img_path)
            dimg = img.copy()

            # load bounding box from file
            bbox_path = img_path[:-4] + "_bbox.txt"
            with open(
                os.path.join(bbox_path),
                "r",
            ) as f:
                # read line and remove the newline character
                bbox = f.readline().rstrip("\n").split(" ")
                bbox = [int(i) for i in bbox]

            bbox = np.array(bbox)

            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = input_size / (max(w, h) * args.bbox_size_scale)
            aimg, M = face_align.transform(img, center, input_size, _scale, rotate)
            aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)

            kps = None
            flips = [0, 1] if USE_FLIP else [0]
            for flip in flips:
                input = aimg.copy()
                if flip:
                    input = input[:, ::-1, :].copy()
                input = np.transpose(input, (2, 0, 1))
                input = np.expand_dims(input, 0)
                imgs = torch.Tensor(input).cuda()
                imgs.div_(255).sub_(0.5).div_(0.5)
                pred = (
                    ldmks_detector(imgs)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                    .reshape((-1, 2))
                )
                pred[:, 0:2] += 1
                pred[:, 0:2] *= input_size // 2
                if flip:
                    pred_flip = pred.copy()
                    pred_flip[:, 0] = input_size - 1 - pred_flip[:, 0]
                    for pair in flip_parts:
                        tmp = pred_flip[pair[0] - 1, :].copy()
                        pred_flip[pair[0] - 1, :] = pred_flip[pair[1] - 1, :]
                        pred_flip[pair[1] - 1, :] = tmp
                    pred = pred_flip
                if kps is None:
                    kps = pred
                else:
                    kps += pred
                    kps /= 2.0

            IM = cv2.invertAffineTransform(M)
            kps = face_align.trans_points(kps, IM)

            # write landmarks to ldmks.txt
            filename = img_path.split("/")[-1]
            with open(
                output_dir + "/%s_ldmks.txt" % filename[:-4],
                "w",
            ) as f:
                for l in range(kps.shape[0]):
                    f.write("%f %f\n" % (kps[l][0], kps[l][1]))

            # render image with landmarks
            kps = kps.astype(int)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)

            cv2.imwrite(render_dir + "/%s" % filename, dimg)

    if args.stage == "dataset":
        # compare the image filenames in dataset_dir and output_dir
        # copy all the detection metadata from output_dir to dataset_dir if an image is in dataset_dir

         copy_detection_metadata(dataset_dir, output_dir)
         process_images(dataset_dir, output_dir, bbox_size_scale=args.bbox_size_scale)
