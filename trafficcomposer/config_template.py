"""
This a template file for the configuration of the TrafficComposer project.
To use this file, please rename it to `config.py` and fill in the required paths.
The DATA_ROOT_DIR should be the path to the root directory of the data.

By default, the architecture of the data directory should be as follows:

DATA_ROOT_DIR
    ├── inputs
    │   ├── textual_description
    │   │   ├── 0.txt
    │   │   ├── 1.txt
    │   │   ├── ...
    │   ├── reference_image
    │   │   ├── 0.jpg
    ├── results
    │   ├── textual
    │   │   ├── gpt-4o
    │   │   │   ├── 0.txt
    │   │   │   ├── 1.txt
    │   │   │   ├── ...
    │   │   ├── ...
    │   ├── ...
    ├── ...
"""

import os


DATA_ROOT_DIR = ...  # The path to the data root directory.
LLAMA3_1_HOME_PATH = ...
LLAMA3_HOME_PATH = ...
LLAMA2_HOME_PATH = ...


IMAGE_DIR = os.path.join(DATA_ROOT_DIR, "inputs/reference_image")

SOURCE_IMAGE_DIR = IMAGE_DIR
VISUAL_IR_LOAD_DIR = os.path.join(DATA_ROOT_DIR, "results/visual/visual_ir")
TEXTUAL_IR_LOAD_DIR = os.path.join(DATA_ROOT_DIR, "results/textual/gpt-4o")
MERGED_IR_SAVE_DIR = os.path.join(DATA_ROOT_DIR, "results/trafficcomposer_aligned_ir")
