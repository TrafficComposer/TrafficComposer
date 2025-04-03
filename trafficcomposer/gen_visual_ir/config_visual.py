import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
print(
    f"Added Project Root to sys.path: {os.path.join(os.path.dirname(__file__), '../..')}"
)

from trafficcomposer.config import DATA_ROOT_DIR

DATA_ROOT_DIR = DATA_ROOT_DIR
IMAGE_LIST_FILE = os.path.join(DATA_ROOT_DIR, "image_list.txt")  # The path to the image list file.
IMAGE_DIR = os.path.join(
    DATA_ROOT_DIR, "inputs/reference_image"
)  # The path to the image directory.

VISUAL_EXP_DIR = os.path.join(
    DATA_ROOT_DIR, "results/visual"
)  # The path to the visual IR experiment directory.

LANE_DETECTION_RESULT_SAVE_DIR = os.path.join(
    VISUAL_EXP_DIR, "lane_detection_results"
)  # The path to the lane detection result directory.
LANE_DETECTION_RESULT_LOAD_DIR = os.path.join(
    LANE_DETECTION_RESULT_SAVE_DIR, "visualization"
)  # The path to the lane detection result directory, which contains the .txt files.

OBJ_DETECTION_RESULT_SAVE_DIR = os.path.join(
    VISUAL_EXP_DIR, "obj_detection_results"
)  # The path to the object detection result directory.
OBJ_DETECTION_RESULT_LOAD_DIR = os.path.join(
    OBJ_DETECTION_RESULT_SAVE_DIR,
    "labels",
)  # The path to the object detection result directory, which contains the .txt files.

VISUAL_IR_SAVE_DIR = os.path.join(
    VISUAL_EXP_DIR, "visual_ir"
)  # The path to the visual IR result directory.
