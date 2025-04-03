"""
How to use this script:
1. Download the pre-trained model and put it in the current directory. By default, we use the model `culane_dla34.pth`
2. Set either one of the two paths. By default, we use the `IMAGE_DIR` to extract lanes from the images in the directory. If user construct the dataset directory as the default setting in `config.py`, the `IMAGE_DIR` should be set to the directory containing the images you want to extract lanes. If user has a txt file containing the paths to the images you want to extract lanes, the `IMAGE_LIST_FILE` should be set to the path to the txt file.
    2.1. The `IMAGE_LIST_FILE` in `config.py` to the path to the txt file containing the paths to the images you want to extract lanes
    2.2. The `IMAGE_DIR` in `config.py` to the directory containing the images you want to extract lanes.
3. Run with the following command:
    python extract_lane.py clrnet/configs/clrnet/clr_dla34_culane.py --load_from culane_dla34.pth --gpus 0
"""

import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner
from clrnet.datasets import build_dataloader
from clrnet.utils.visualization import imshow_lanes
from clrnet.datasets.process import Process
from clrnet.datasets.culane import CULane

from tqdm import tqdm

import copy
import logging


import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# My configuration
from trafficcomposer.gen_visual_ir.config_visual import IMAGE_DIR, LANE_DETECTION_RESULT_SAVE_DIR
from trafficcomposer.gen_visual_ir.config_visual import OBJ_DETECTION_RESULT_SAVE_DIR

# from config import IMAGE_LIST_FILE


from utils import gen_img_list, copy_dir


def convert_dict_to_list(dictionary):
    """
    dictionary:
        {
            key_0: [value_0, value_1, ...],
            key_1: [value_0, value_1, ...],
            key_2: {
                key_2_0: [value_0, value_1, ...],
                key_2_1: [value_0, value_1, ...],
            }
            ...
        }
    return:
        [
            {key_0: value_0, key_1: value_0, ...},
            {key_0: value_1, key_1: value_1, ...},
            ...
        ]
    """
    keys = list(dictionary.keys())
    num_items = 0
    for key in keys:
        num_items = max(len(dictionary[key]), num_items)

    result = []
    for i in range(num_items):
        item = {}
        for key in keys:
            if isinstance(dictionary[key], list):
                item[key] = dictionary[key][i]
            else:
                item[key] = dict(
                    [
                        (sub_key, dictionary[key][sub_key][i])
                        for sub_key in dictionary[key].keys()
                    ]
                )
        result.append(item)

    return result


def transform_lane(lane, lane_img_h, lane_img_w, target_img_h, target_img_w):
    """
    Transform the lane from the original size to the size of the input of the model
    """
    # sample_y = cfg.sample_y
    # ys = np.array(sample_y) / float(img_h)
    # xs = lane(ys)
    # valid_mask = (xs >= 0) & (xs < 1)
    # lane_xs = xs[valid_mask] * img_w
    # lane_ys = ys[valid_mask] * img_h
    # lane = np.concatenate(
    # (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1
    # )
    lane[:, 0] = lane[:, 0] * target_img_w / lane_img_w
    lane[:, 1] = lane[:, 1] * target_img_h / lane_img_h
    return lane


class MyInferDataset(CULane):
    """
    Although this class inherits from CULane, it overrides all the possibly used methods in inference.
    The only used attribute from the parent class is the processes from BaseDataset, which is the parent class of CULane
    """

    def __init__(self, image_list_file=None, image_dir=None, cfg=None):
        """
        Args:
            image_list_file: str, the path to the txt file containing the paths to the images you want to extract lanes
        """
        # The following attributes are not used in this class, legacy arguments from the parent class CULane __init__
        data_root = None
        split = "test"

        self.split = split

        # The following are from BaseDataset __init__, which CULane also calls
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = "train" in split

        processes = cfg.val_process
        self.processes = Process(processes, cfg)

        # self.list_path = os.path.join(data_root, LIST_FILE[split])
        self.list_path = image_list_file

        # self.load_annotations()

        if image_list_file is not None:
            assert (
                image_dir is None
            ), "Only one of image_list_file and image_dir should be provided."
            self.logger.info(f"Build dataset using image_list_file: {image_list_file}")
            self.load_annotations()
        elif image_dir is not None:
            assert (
                image_list_file is None
            ), "Only one of image_list_file and image_dir should be provided."
            self.logger.info(f"Build dataset using image_dir: {image_dir}")
            self.load_annotations_from_dir(image_dir)
        else:
            raise ValueError("Either image_list_file or image_dir should be provided.")

    def load_annotations_from_dir(self, image_dir):
        self.logger.info("Loading dataset...")

        ls_img = gen_img_list(image_dir)

        self.data_infos = []
        for line in ls_img:
            infos = self.load_annotation([line])
            self.data_infos.append(infos)

    def load_annotations(self):
        self.logger.info("Loading dataset...")

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

    def load_annotation(self, line):
        """
        Collect the information of the image path and image name.
        Previous CLRNet includes the lane annotations in the CULane dataset.
        We don't have the lane annotations in our inference setting.
        """
        infos = {}
        img_line = line[0]
        # img_line = img_line[1 if img_line[0] == "/" else 0 : :]

        # img_path = os.path.join(self.data_root, img_line)
        img_path = img_line

        infos["img_name"] = img_line.split("/")[-1]
        infos["img_path"] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == "/" else 0 : :]
            mask_path = os.path.join(self.data_root, mask_line)
            infos["mask_path"] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos["lane_exist"] = np.array(exist_list)

        infos["lanes"] = []
        #
        # ## The following code is not used in this class. It was used to load the lane annotations in CULane.
        # ## In our inference setting, we do not have the lane annotations ground truth.
        #
        # anno_path = img_path[:-3] + "lines.txt"  # remove sufix jpg and add lines.txt
        # with open(anno_path, "r") as anno_file:
        #     data = [list(map(float, line.split())) for line in anno_file.readlines()]
        # lanes = [
        #     [
        #         (lane[i], lane[i + 1])
        #         for i in range(0, len(lane), 2)
        #         if lane[i] >= 0 and lane[i + 1] >= 0
        #     ]
        #     for lane in data
        # ]
        # lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        # lanes = [
        #     lane for lane in lanes if len(lane) > 2
        # ]  # remove lanes with less than 2 points

        # lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        # infos["lanes"] = lanes

        return infos

    def view(self, predictions, img_metas):
        """
        img_metas:
            original: List of dictionaries, each containing the metadata of an image
            img_metas = [{"full_img_path": str, "img_name": str, "img_ori_shape": dict}, ...]
        """
        # img_metas = [item for img_meta in img_metas.data for item in img_meta]

        img_metas = convert_dict_to_list(img_metas)

        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta["img_name"]
            img = cv2.imread(img_meta["full_img_path"])
            ori_img_h, ori_img_w = img.shape[:2]
            img_save_path = os.path.join(
                self.cfg.work_dir, "visualization", img_name.replace("/", "_")
            )

            # print("========== In ImageDataset.view ==========")
            # print(f"img_save_path: {img_save_path}")
            # print(f"lanes: {lanes}")

            # ## the lane in lanes is based on the image size being cfg.ori_img_w, cfg.ori_img_h
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            # ## To accommodate different image sizes, we need to transform the lanes according to the original image size
            lanes = [
                transform_lane(
                    lane,
                    lane_img_h=self.cfg.ori_img_h,
                    lane_img_w=self.cfg.ori_img_w,
                    target_img_h=ori_img_h,
                    target_img_w=ori_img_w,
                )
                for lane in lanes
            ]

            # print(f"========== After to_array ==========")
            # print(f"lanes: {lanes}")

            imshow_lanes(img, lanes, out_file=img_save_path)

            output_str = self.convert_lane_np_to_txt(lanes)
            with open(img_save_path + ".lines.txt", "w") as f:
                f.write(output_str)

    def convert_lane_np_to_txt(self, lanes):
        """
        convert the lanes to a txt file
        """
        out = []
        for lane in lanes:
            lane_xs = lane[:, 0]
            lane_ys = lane[:, 1]
            lane_str = " ".join(
                ["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)]
            )
            if lane_str != "":
                out.append(lane_str)

        return "\n".join(out)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info["img_path"])

        # ## You have to resize the image to the size of the input of the model
        # ## Otherwise, the model will not be able to detect the lanes
        img = cv2.resize(img, (self.cfg.ori_img_w, self.cfg.ori_img_h))
        img = img[self.cfg.cut_height :, :, :]
        sample = data_info.copy()
        sample.update({"img": img})

        sample = self.processes(sample)

        # ## ========== DEBUG ==========
        # img_save_path = os.path.join(DEBUG_IMAGE_DIR, data_info["img_name"].replace("/", "_"))
        # img2save = copy.deepcopy(sample["img"])
        # img2save = img2save.numpy().transpose(1, 2, 0)
        # img2save = (img2save * 255).astype(np.uint8)
        # cv2.imwrite(img_save_path, img2save)
        # ## ===========================

        meta = {
            "full_img_path": data_info["img_path"],
            "img_name": data_info["img_name"],
        }
        # meta = DC(meta, cpu_only=True)
        sample.update({"meta": meta})

        return sample


class MyClrnetRunner(Runner):
    """
    To infer the lane detection results on a new image
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def infer(self, image_list_file=None, image_dir=None, result_dir=None):
        """
        To infer the lane detection results on customized images
        """
        batch_size = 24

        dataset = MyInferDataset(
            image_list_file=image_list_file,
            image_dir=image_dir,
            cfg=self.cfg,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            # batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=False,
            drop_last=False,
            # collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            # worker_init_fn=init_fn
        )
        self.dataloader = dataloader

        self.net.eval()
        for _, data in enumerate(tqdm(self.dataloader, desc=f"Inferencing")):
            data = self.to_cuda(data)

            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
            # if self.cfg.view:
            self.dataloader.dataset.view(output, data["meta"])

        copy_dir(self.cfg.work_dir, result_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dirs", type=str, default=None, help="work dirs")
    parser.add_argument(
        "--load_from", default=None, help="the checkpoint file to load from"
    )
    parser.add_argument(
        "--resume_from", default=None, help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--finetune_from", default=None, help="the checkpoint file to resume from"
    )
    parser.add_argument("--view", action="store_true", help="whether to view")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to test the checkpoint on testing set",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default="0")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    runner = MyClrnetRunner(cfg)

    # runner.infer(image_list_file=IMAGE_LIST_FILE, image_dir=None)
    runner.infer(
        image_list_file=None,
        image_dir=IMAGE_DIR,
        # image_dir=OBJ_DETECTION_RESULT_SAVE_DIR,
        result_dir=LANE_DETECTION_RESULT_SAVE_DIR,
    )


if __name__ == "__main__":
    main()

    # from config import IMAGE_DIR
    # gen_img_list(IMAGE_DIR, IMAGE_LIST_FILE)
