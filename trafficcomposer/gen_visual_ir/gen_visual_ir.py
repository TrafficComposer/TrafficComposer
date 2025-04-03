import os
import cv2
from tqdm import tqdm
import yaml
import json
import copy

from pprint import pprint

import sys

sys.path.append("../..")

from trafficcomposer.gen_visual_ir.config_visual import (
    IMAGE_DIR,
    LANE_DETECTION_RESULT_LOAD_DIR,
    OBJ_DETECTION_RESULT_LOAD_DIR,
    VISUAL_IR_SAVE_DIR,
)
from utils import is_image_file


class VisualIRGenerator(object):
    """
    To generate the visual IR from both the lane detection results and the object detection results
    """

    def __init__(
        self,
        source_img_dir=IMAGE_DIR,
        lane_detection_dir=LANE_DETECTION_RESULT_LOAD_DIR,
        obj_detection_dir=OBJ_DETECTION_RESULT_LOAD_DIR,
        save_dir=VISUAL_IR_SAVE_DIR,
    ):
        assert os.path.isdir(source_img_dir), f"{source_img_dir} is not a directory."
        assert os.path.isdir(
            lane_detection_dir
        ), f"{lane_detection_dir} is not a directory."
        assert os.path.isdir(
            obj_detection_dir
        ), f"{obj_detection_dir} is not a directory."

        self.source_img_dir = source_img_dir
        self.lane_detection_dir = lane_detection_dir
        self.obj_detection_dir = obj_detection_dir

        with open("coco.yaml", "r") as f:
            self.dct_coco = yaml.load(f, Loader=yaml.FullLoader)["names"]

        self.save_dir = save_dir
        self.build_save_dir(save_dir)

        self.debug = True

    def build_save_dir(self, save_dir):
        if os.path.exists(save_dir):
            # If the save_dir exists, delete it and create a new one.
            input(
                f"WARNING: {save_dir} already exists. Press Enter to delete it and create a new one."
            )
            os.system(f"rm -rf {save_dir}")

        # If the save_dir does not exist, simply create it.
        os.makedirs(save_dir)
        print(f"Save dir {save_dir} is created.")

    def visualize_obj_bbox(self, img, coord_xyxy):
        # Note that the coordinates are in (x, y, x_len, y_len) format
        # (x, y) is the upper left corner of the rectangle
        cv2.rectangle(
            img,
            (
                coord_xyxy[0],
                coord_xyxy[1],
                coord_xyxy[2] - coord_xyxy[0],
                coord_xyxy[3] - coord_xyxy[1],
            ),
            (0, 255, 0),
            2,
        )
        cv2.circle(img, (coord_xyxy[0], coord_xyxy[1]), 5, (0, 0, 255), -1)
        cv2.circle(img, (coord_xyxy[2], coord_xyxy[3]), 5, (0, 0, 255), -1)
        # cv2.imwrite(f"tmp.png", img)
        return img

    @staticmethod
    def visualize_pt(img, x, y):
        # cv2.rectangle(
        #     img,
        #     (coord_xyxy[0], coord_xyxy[1], coord_xywh[2], coord_xywh[3]),
        #     (0, 255, 0),
        #     2,
        # )
        cv2.circle(img, (x, y), 50, (0, 0, 255), -1)
        # cv2.circle(img, (coord_xyxy[2], coord_xyxy[3]), 50, (0, 0, 255), -1)
        # cv2.imwrite(f"tmp.png", img)
        return img

    def assign_actor2lane(self, coord_left_bottom, coord_right_bottom, ls_lines):
        """
            Assign the lane index for the vehicle or pedestrian

                line idx
        lane idx   0   lane idx   1      2        3
           -1      |      0       |   1   |   2   |    3
              *****|*****         |       |       |
             *     |   *          |       |       |
            *      |  *           |       |       |
           ********|**            |       |       |
                   |              |       |       |
            :param coord_left_bottom: tuple, the coordinates of the left bottom corner of the bbox of the vehicle in the format of (x, y)
            return: int, the index of the lane the vehicle belongs to
        """

        # xyxy: the coordinates of the bbox of the vehicle in the format of (x1, y1, x2, y2)
        # coord_left_bottom = (xyxy[0], xyxy[3])
        # coord_right_bottom = (xyxy[2], xyxy[3])

        threshold_area = 0.6

        line_idx_on_point_right = None
        left_line_anchor = (None, None)
        for line_idx, line in enumerate(ls_lines):
            # iterate from the left to the right
            is_on_left, left_line_anchor = self.is_point_on_left(
                coord_left_bottom, line
            )
            if is_on_left:
                line_idx_on_point_right = line_idx
                break

        if line_idx_on_point_right is None:
            # The vehicle is on the right side of the rightmost lane
            return len(ls_lines) - 1

        line_idx_on_point_left = None
        right_line_anchor = (None, None)
        for line_idx in range(len(ls_lines) - 1, -1, -1):
            # iterate from the right to the left
            is_on_right, right_line_anchor = self.is_point_on_right(
                coord_right_bottom, ls_lines[line_idx]
            )
            if is_on_right:
                line_idx_on_point_left = line_idx
                break

        if line_idx_on_point_left is None:
            # The vehicle is on the left side of the leftmost lane
            return -1

        if (
            line_idx_on_point_right == line_idx_on_point_left
            and left_line_anchor[1] is not None
        ):
            # ## The vehicle is on the lane dividing line line_idx_on_point_right.
            # In this case, we need to decide which lane the vehicle belongs to.
            if left_line_anchor[1] - coord_left_bottom[1] > (
                coord_right_bottom[1] - left_line_anchor[1]
            ):
                """
                        line idx
                lane idx   0   lane idx
                   -1      |      0
                      *****|******
                     *     |    *
                    *      |   *
                   ******* @ **
                           | @ anchor point
                """
                return line_idx_on_point_right - 1
            elif left_line_anchor[1] - coord_left_bottom[1] <= (
                coord_right_bottom[1] - left_line_anchor[1]
            ):
                return line_idx_on_point_right
        else:
            if line_idx_on_point_left - line_idx_on_point_right == 1:
                """
                        line_idx_on_point_right
                lane idx   0   lane idx
                   -1      |      0     | line_idx_on_point_left (1)
                      *****|************|*******
                     *     |            |     *
                    *      |            |    *
                   ******* @ ***********|****
                           | @ anchor point
                """
                return line_idx_on_point_right
            elif line_idx_on_point_right - line_idx_on_point_left == 1:
                """
                        line_idx_on_point_left (0)
                lane idx   0   lane idx
                   -1      |      0     | line_idx_on_point_right (1)
                           |            |
                           |     *******|*
                           |    *       *
                           |   *       *|
                           @  ********* |
                           | @ anchor point
                """
                return line_idx_on_point_left
            elif line_idx_on_point_left - line_idx_on_point_right > 1:
                return line_idx_on_point_right + 1

    @staticmethod
    def is_point_on_left(coord, ls_pt_in_line):
        """
        ls_pt_in_line: list of tuples, [(x1, y1), (x2, y2), ...]
            y1 > y2 > y3 > ...

        Check whether the point is on the left of the line
        """
        x, y = coord
        x_line_anchor, y_line_anchor = None, None
        for idx, (x_line, y_line) in enumerate(ls_pt_in_line):
            # ## Step 1. Search for the point in the line with the same y coordinate
            if y == y_line:
                x_line_anchor = x_line
            elif y_line > y:
                continue
            elif y_line <= y:
                # ## Found the anchor point
                x_line_anchor = x_line
                y_line_anchor = y_line
                break
        if x_line_anchor is None:
            print("WARNING: Cannot find the anchor point")
            if x <= x_line:
                return True, (x_line, y_line)
            elif x > x_line:
                return False, (x_line, y_line)
        if x <= x_line_anchor:
            return True, (x_line_anchor, y_line_anchor)
        elif x > x_line_anchor:
            return False, (x_line_anchor, y_line_anchor)

    @staticmethod
    def is_point_on_right(coord, ls_pt_in_line):
        """
        ls_pt_in_line: list of tuples, [(x1, y1), (x2, y2), ...]
            y1 > y2 > y3 > ...
        Check whether the point is on the right of the line
        """
        x, y = coord
        x_line_anchor, y_line_anchor = None, None
        for idx, (x_line, y_line) in enumerate(ls_pt_in_line):
            # ## Step 1. Search for the point in the line with the same y coordinate
            if y == y_line:
                x_line_anchor = x_line
                y_line_anchor = y_line
            elif y_line > y:
                continue
            elif y_line <= y:
                # ## Found the anchor point
                x_line_anchor = x_line
                y_line_anchor = y_line
                break
        # ## Step 2. Check whether the point is on the right of the line
        if idx == 0:
            # ## The point in the line at the bottom of the image is higher than the coord point
            print(
                "WARNING: The `coord` point at the edge of the image. Cannot find the anchor point"
            )
            if ls_pt_in_line[0][1] < y:
                return True, (coord[0], coord[1])
        if x_line_anchor is None:
            print("WARNING: Cannot find the anchor point")
            if x >= x_line:
                return True, (x_line, y_line)
            elif x < x_line:
                return False, (x_line, y_line)
        if x >= x_line_anchor:
            return True, (x_line_anchor, y_line_anchor)
        elif x < x_line_anchor:
            return False, (x_line_anchor, y_line_anchor)

    def gen_visual_ir(
        self, source_img_path, lane_detection_file_name, obj_detection_file_name
    ):
        """
        yolo_item: str following the format "[class] [x_center_normalized] [y_center_normalized] [width_normalized] [height_normalized]"
        0/0 -----> x axis
        |
        |
        v
        y axis
        """
        img = cv2.imread(source_img_path)
        img_h, img_w = img.shape[:2]

        # ## load the lane detection results
        with open(
            os.path.join(self.lane_detection_dir, lane_detection_file_name), "r"
        ) as f:
            ls_lines_raw = f.readlines()

        # ## extract all the lines
        ls_lines = []
        for line in ls_lines_raw:
            line = line.strip("\n")
            ls_line = line.split(" ")

            ls_line = [int(float(x)) for x in ls_line]
            ls_line_coor = list(zip(ls_line[0::2], ls_line[1::2]))
            ls_lines.append(ls_line_coor)

        # ## rank the lines from left to right
        ls_lines.sort(key=lambda x: x[0][0])

        # ## load the actor detection results
        with open(
            os.path.join(self.obj_detection_dir, obj_detection_file_name), "r"
        ) as f:
            ls_actors_raw = f.readlines()
        ls_actors_raw = [x.strip("\n") for x in ls_actors_raw]

        # ## assign lane for each actor
        dct_lane2actor = {}  # {lane_idx: [actor_yolo_item]}
        for actor_yolo_item in ls_actors_raw:
            # ## actor_yolo_item: str following the format "[class] [x_center_normalized] [y_center_normalized] [width_normalized] [height_normalized]"
            single_actor_info_ls = actor_yolo_item.split(" ")

            xywhn = (
                float(single_actor_info_ls[1]),
                float(single_actor_info_ls[2]),
                float(single_actor_info_ls[3]),
                float(single_actor_info_ls[4]),
            )
            xyxyn = (
                xywhn[0] - xywhn[2] / 2,
                xywhn[1] - xywhn[3] / 2,
                xywhn[0] + xywhn[2] / 2,
                xywhn[1] + xywhn[3] / 2,
            )
            xyxy = (
                int(xyxyn[0] * img_w),
                int(xyxyn[1] * img_h),
                int(xyxyn[2] * img_w),
                int(xyxyn[3] * img_h),
            )

            coord_left_bottom = (xyxy[0], xyxy[3])
            coord_right_bottom = (xyxy[2], xyxy[3])

            if self.debug:
                tmp_img = copy.deepcopy(img)
                tmp_img = self.visualize_obj_bbox(tmp_img, xyxy)
                cv2.imwrite("tmp.png", tmp_img)
                print(f"Processing Object: {actor_yolo_item}")

            # ## assign the lane index for the vehicle
            lane_idx = self.assign_actor2lane(
                coord_left_bottom, coord_right_bottom, ls_lines
            )

            if lane_idx is not None:
                # ## assign vehicle to the lane
                if lane_idx not in dct_lane2actor:
                    dct_lane2actor[lane_idx] = [actor_yolo_item]
                else:
                    dct_lane2actor[lane_idx].append(actor_yolo_item)
            else:
                print("Cannot find the lane for the vehicle")
                # ## for visualization
                if self.debug:
                    cv2.circle(img, coord_left_bottom, 50, (0, 255, 0), -1)
                    cv2.circle(img, coord_right_bottom, 50, (0, 255, 0), -1)
                    cv2.imwrite("tmp.png", img)
                    input("Press Enter to continue...")

        # ## Assign lane to ego vehicle
        x_middle = img_w // 2
        y_bottom = img_h
        ego_vehicle_width = img_w // 10
        ego_left_bottom = (x_middle - ego_vehicle_width // 2, y_bottom)
        ego_right_bottom = (x_middle + ego_vehicle_width // 2, y_bottom)
        ego_line_idx = self.assign_actor2lane(
            ego_left_bottom, ego_right_bottom, ls_lines
        )

        # # # ## Select the lane with the maximum horizontal line
        # # left_line = ls_lines[0]
        # # left_line_y_max = max([y for x, y in left_line])
        # # right_lines = ls_lines[-1]
        # # right_line_y_max = max([y for x, y in right_lines])
        # # line_y_max = max(left_line_y_max, right_line_y_max)

        # ## Post-processing
        # ## assign vehicle type to each vehicle
        # print(dct_lane_vehicle)
        dct_lane2actor_clsnm = (
            {}
        )  # {lane_idx: [(actor_class_name_in_coco_dataset, actor_yolo_item), ...]}
        for lane in list(dct_lane2actor.keys()):
            dct_lane2actor_clsnm[lane] = []
            for actor_yolo_item in dct_lane2actor[lane]:
                actor_class_name = self.dct_coco[int(actor_yolo_item.split(" ")[0])]
                dct_lane2actor_clsnm[lane].append((actor_class_name, actor_yolo_item))

        if ego_line_idx in dct_lane2actor_clsnm:
            dct_lane2actor_clsnm[ego_line_idx].append(("ego", "ego 0 0 0 0"))
        else:
            dct_lane2actor_clsnm[ego_line_idx] = [("ego", "ego 0 0 0 0")]

        #     cv2.imwrite("tmp.png", img)

        print(dct_lane2actor_clsnm)
        return dct_lane2actor_clsnm

    def main(self):
        """
        The main function to generate_visual_ir.
        Itereate over all the images in the source_img_dir, locte the corresponding lane detection results and object detection results, and assign the lane index for each vehicle.
        """

        ls_source_img_names = os.listdir(self.source_img_dir)
        ls_source_img_names = [x for x in ls_source_img_names if is_image_file(x)]
        ls_source_img_names.sort()

        # ## load the lane detection results name list
        ls_lane_detection_files = os.listdir(self.lane_detection_dir)
        ls_lane_detection_files = [
            x for x in ls_lane_detection_files if x.endswith(".lines.txt")
        ]
        ls_lane_detection_files.sort()

        # ## load the object detection results name list
        ls_obj_detection_files = os.listdir(self.obj_detection_dir)
        ls_obj_detection_files = [
            x for x in ls_obj_detection_files if x.endswith(".txt")
        ]
        ls_obj_detection_files.sort()

        for idx, source_img_name in enumerate(tqdm(ls_source_img_names)):
            source_img_name_no_ext = ".".join(source_img_name.split(".")[:-1])
            source_img_path = os.path.join(self.source_img_dir, source_img_name)

            # ## locate the lane detection results
            lane_detection_file_name = ls_lane_detection_files[idx]
            assert lane_detection_file_name.startswith(
                f"{source_img_name_no_ext}."
            ), f"{lane_detection_file_name} does not match {source_img_name}"

            # ## locate the actor detection results
            obj_detection_file_name = ls_obj_detection_files[idx]
            assert obj_detection_file_name.startswith(
                f"{source_img_name_no_ext}."
            ), f"{obj_detection_file_name} does not match {source_img_name}"

            visual_ir = self.gen_visual_ir(
                source_img_path, lane_detection_file_name, obj_detection_file_name
            )

            # ## Save results to YAML files
            with open(
                os.path.join(self.save_dir, f"{source_img_name_no_ext}.yaml"), "w"
            ) as f:
                yaml.dump(visual_ir, f)

            # # ## Save results to JSON files
            # with open(os.path.join(self.save_dir, f"{source_img_name}.json"), "w") as f:
            #     json.dump(visual_ir, f)


if __name__ == "__main__":
    from trafficcomposer.gen_visual_ir.config_visual import (
        OBJ_DETECTION_RESULT_SAVE_DIR,
    )

    runner = VisualIRGenerator(
        # source_img_dir=IMAGE_DIR,
        source_img_dir=OBJ_DETECTION_RESULT_SAVE_DIR,
        lane_detection_dir=LANE_DETECTION_RESULT_LOAD_DIR,
        obj_detection_dir=OBJ_DETECTION_RESULT_LOAD_DIR,
        save_dir=VISUAL_IR_SAVE_DIR,
    )
    runner.debug = False
    runner.main()

    # str_ret = runner.assign_vehicle_lane(is_debug=True)
    # print(f"assign_vehicle_lane: {str_ret}")
    # with open("case_img.yaml", "w") as f:
    #     yaml.dump(str_ret, f)
