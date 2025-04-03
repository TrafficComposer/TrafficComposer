import os
# import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# print(
#     f"Added Project Root to sys.path: {os.path.join(os.path.dirname(__file__), '../..')}"
# )

from trafficcomposer.config import DATA_ROOT_DIR

DATA_ROOT_DIR = DATA_ROOT_DIR

SOURCE_IMAGE_DIR = os.path.join(DATA_ROOT_DIR, "inputs/reference_image")

DESCRIPTION_DIR = os.path.join(DATA_ROOT_DIR, "inputs/textual_description")

BASELINE_GPT4O_SAVE_PATH = os.path.join(DATA_ROOT_DIR, "results/baselines/gpt-4o")

TRAIN_IMAGE_EG1_PATH = os.path.join(os.path.dirname(__file__), "example_imgs/eg1.png")
TRAIN_IMAGE_EG2_PATH = os.path.join(os.path.dirname(__file__), "example_imgs/eg2.png")
