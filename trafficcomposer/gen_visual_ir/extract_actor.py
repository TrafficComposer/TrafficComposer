"""
Command: python extract_actor.py
---------------------------------
This script is used to extract the actor from the image using YOLOv10 model.
The model is pretrained on the COCO dataset.
The input images are stored in the IMAGE_DIR.
The output images are stored in the OBJ_DETECTION_RESULT_SAVE_DIR.
"""

from ultralytics import YOLOv10

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from trafficcomposer.gen_visual_ir.config_visual import (
    IMAGE_DIR,
    LANE_DETECTION_RESULT_LOAD_DIR,
    OBJ_DETECTION_RESULT_SAVE_DIR,
)

from utils import gen_img_list, copy_dir


def yolo_detect(img_dir=IMAGE_DIR):
    model = YOLOv10.from_pretrained("jameslahm/yolov10x")

    # img_list = gen_img_list(img_dir)
    results = model.predict(source=img_dir, save=True, save_txt=True)

    copy_dir(results[0].save_dir, OBJ_DETECTION_RESULT_SAVE_DIR)


if __name__ == "__main__":
    yolo_detect(img_dir=IMAGE_DIR)
