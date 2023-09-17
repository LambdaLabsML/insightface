from __future__ import division
import numpy as np
import os
import os.path as osp
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

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(224, 224))


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file=None, session=None):
        import onnxruntime

        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in scrfd model, ignore")
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                # solution-3:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.5, input_size=None, max_num=0, metric="default"):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (
                    area - offset_dist_squared * 2.0
                )  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


def get_scrfd(name, download=False, root="~/.insightface/models", **kwargs):
    if not download:
        assert os.path.exists(name)
        return SCRFD(name)
    else:
        from .model_store import get_model_file

        _file = get_model_file("scrfd_%s" % name, root=root)
        return SCRFD(_file)


def scrfd_2p5gkps(**kwargs):
    return get_scrfd("2p5gkps", download=True, **kwargs)


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
    parser.add_argument("--bbox_size_scale", type=float, default=2.5)
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
                x1, y1, x2, y2 = bbox.astype(np.int_)

                w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
                center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
                rotate = 0
                _scale = input_size / (max(w, h) * 1.5)
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

                    x1, y1, x2, y2, score = bbox.astype(np.int_)

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
            _scale = input_size / (max(w, h) * 1.5)
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
            kps = kps.astype(np.int)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)

            cv2.imwrite(render_dir + "/%s" % filename, dimg)

    if args.stage == "dataset":
        # compare the image filenames in dataset_dir and output_dir
        # copy all the detection metadata from output_dir to dataset_dir if an image is in dataset_dir

        img_paths = glob.glob(os.path.join(dataset_dir, "*.png"))

        for img_path in img_paths:
            filename = img_path.split("/")[-1]
            source_filename = output_dir + "/" + filename
            if os.path.exists(source_filename):
                bbox_path = source_filename.replace(".png", "_bbox.txt")
                ldmks_path = source_filename.replace(".png", "_ldmks.txt")
                shutil.copy(bbox_path, dataset_dir)
                shutil.copy(ldmks_path, dataset_dir)
                print("cp meta files:", filename)

        X = []
        Y = []

        for img_path in img_paths:
            img = cv2.imread(img_path)
            dimg = img.copy()
            ylines = open(img_path.replace(".png", "_ldmks.txt")).readlines()
            ylines = ylines[:68]
            y = []
            for yline in ylines:
                lmk = [float(x) for x in yline.strip().split()]
                y.append(tuple(lmk))
            pred = np.array(y)

            # load bounding box from file
            bbox_path = img_path.replace(".png", "_bbox.txt")
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
            _scale = dataset_output_size / (max(w, h) * 1.5)
            aimg, M = face_align.transform(
                dimg, center, dataset_output_size, _scale, rotate
            )
            pred = face_align.trans_points(pred, M)

            x = img_path.split("/")[-1]
            x = x.replace("png", "jpg")
            X.append(x)

            y = []
            for k in range(pred.shape[0]):
                y.append((pred[k][0], pred[k][1]))
            Y.append(y)
            cv2.imwrite("%s/%s" % (dataset_dir, x), aimg)

        with open(osp.join(dataset_dir, "annot.pkl"), "wb") as pfile:
            pickle.dump((X, Y), pfile, protocol=pickle.HIGHEST_PROTOCOL)
