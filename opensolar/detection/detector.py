import inspect
import os
import sys

import cv2
import numpy as np
import torch
from opensolar.detection.models.common import DetectMultiBackend

from opensolar.detection.utils.augmentations import letterbox
from opensolar.detection.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    scale_segments,
)
from opensolar.detection.utils.plots import Annotator, colors, save_one_box
from opensolar.detection.utils.segment.general import (
    masks2segments,
    process_mask,
    process_mask_native,
    scale_image,
)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from optimizer.solver import place_solar_panels


class Detector:
    def __init__(
        self, conf_thres=0.45, iou_thres=0.7, weight_path="", imgsz=(640, 640)
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.model = DetectMultiBackend(
            weight_path, device="cpu", data="./data/rooftop.yaml"
        )
        self.stride = self.model.stride
        self.names = self.model.names

    def preprocess_image(self, im, img_size, stride, auto):
        im = letterbox(im, img_size, stride=stride, auto=auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        return im

    def detect(self, img):
        instances = []

        img_ori = img.copy()
        ori_img_shape = img.shape

        img = self.preprocess_image(img, self.imgsz, self.stride, True)
        img_shape = img.shape

        img = img[None]
        img = torch.tensor(img) / 255.0

        pred, proto = self.model(img)[:2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, nm=32)

        det = pred[0]

        pred_masks = process_mask(
            proto[0], det[:, 6:], det[:, :4], img_shape[1:], upsample=True
        )  # HWC
        det[:, :4] = scale_boxes(
            img_shape[1:], det[:, :4], ori_img_shape
        ).round()  # rescale boxes to im0 size

        for bbox, mask in zip(det, pred_masks):
            mask = mask.numpy()
            mask = (scale_image(img_shape[1:], mask, ori_img_shape) * 255).astype(np.uint8)[:, :, 0]
            instances.append({
                'cls': self.names[int(bbox[5])],
                'cls_id': int(bbox[5]),
                'bbox': bbox[:-4].numpy().astype(np.int32),
                'mask': mask
            })
        
        meta_info = {
            'img_scaled': img[0],
            'orig_shape': ori_img_shape,
            'pred_masks': pred_masks
        }

        return meta_info, instances


if __name__ == "__main__":
    model = Detector(conf_thres=0.45, iou_thres=0.7, weight_path="./weights/best.pt")
    img = cv2.imread("./e13e1236-5136-478a-b068-59fcd4ea61e3.jpeg")
    img, instances = model.detect(img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    place_solar_panels(instances, img)