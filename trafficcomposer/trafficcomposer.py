import os
from tqdm import tqdm
import json
import yaml
import copy

from pprint import pprint


from config import (
    SOURCE_IMAGE_DIR,
    TEXTUAL_IR_LOAD_DIR,
    VISUAL_IR_LOAD_DIR,
    MERGED_IR_SAVE_DIR,
)
from gen_visual_ir.utils import is_image_file


class TrafficComposer:
    def __init__(
        self,
        source_image_dir=SOURCE_IMAGE_DIR,
        textual_ir_dir=TEXTUAL_IR_LOAD_DIR,
        visual_ir_dir=VISUAL_IR_LOAD_DIR,
        save_dir=MERGED_IR_SAVE_DIR,
    ):
        self.debug = False

        self.source_img_dir = source_image_dir
        self.textual_ir_dir = textual_ir_dir
        self.visual_ir_dir = visual_ir_dir
        self.save_dir = save_dir

    def align_two_modalities(self, textual_ir_fp, visual_ir_fp):
        """
        Merge the textual ir and the visual ir
        """
        dct_direction = {
            "left": "left",
            "right": "right",
            "ahead": "ahead",
            "in front": "ahead",
            "behind": "behind",
        }

        # ## load the Textual IR
        assert isinstance(textual_ir_fp, str)
        if type(textual_ir_fp) == str:
            os.path.exists(textual_ir_fp), f"{textual_ir_fp} does not exist"
            with open(textual_ir_fp, "r") as f:
                txt = f.read()
                dct_text = yaml.safe_load(txt)
            if type(dct_text) == str:
                dct_text = yaml.load(dct_text, Loader=yaml.FullLoader)
        else:
            dct_text = copy.copy(textual_ir_fp)

        # ## load the Visual IR
        assert isinstance(visual_ir_fp, str)
        if type(visual_ir_fp) == str:
            os.path.exists(visual_ir_fp), f"{visual_ir_fp} does not exist"
            with open(visual_ir_fp, "r") as f:
                dct_visual = yaml.load(f, Loader=yaml.FullLoader)
        else:
            dct_visual = copy.copy(visual_ir_fp)

        assert (
            dct_text is not None and dct_visual is not None
        ), "Neither IR should be None"

        if self.debug:
            print("dct_text:")
            pprint(dct_text)
            print()
            print(f"dct_visual:")
            pprint(dct_visual)
            print()
            print(f"type(dct_text): {type(dct_text)}")
            print(f"type(dct_visual): {type(dct_visual)}")

        # ## register all the detected vehicles in dct_visual
        dct_vehicle = {}
        for lane in dct_visual.keys():
            dct_vehicle[lane] = []
            for vehicle in dct_visual[lane]:
                dct_vehicle[lane].append(False)

        # ## Start from copying the dct_text
        dct_aligned = copy.copy(dct_text)

        if self.debug:
            print(f"dct_aligned: {dct_aligned}")
            print()

        # ## Register the lane number
        dct_aligned["lane_number"] = len(list(dct_visual.keys()))

        # ## Search for the ego vehicle
        # ## To find the lane index of the ego vehicle
        for idx_lane in sorted(list(dct_visual.keys())):
            # print(f"idx_lane: {idx_lane}")
            for idx_vehicle, vehicle in enumerate(dct_visual[idx_lane]):
                if "ego" == vehicle[0]:
                    dct_aligned["participant"]["ego_vehicle"]["lane_idx"] = idx_lane
                    ego_lane_idx = idx_lane

                    dct_vehicle[idx_lane][idx_vehicle] = True
                    break

        for pariticipant in dct_text["participant"]:

            if self.debug:
                print(f"participant: {pariticipant}")

            if pariticipant == "ego_vehicle":
                continue
            else:
                # ## look for the other vehicles
                # ## To find the lane index of the ego vehicle
                # ## According to the vehicle position
                # ## Assumption: the vehicle is the nearest one to the ego vehicle in the direction
                direction = None
                for pos_direction in dct_text["participant"][pariticipant][
                    "position_target"
                ]:
                    if pos_direction in dct_direction.keys():
                        direction = dct_direction[pos_direction]
                        break
                if self.debug:
                    if direction is None:
                        print("Cannot find the direction")
                        input("Press Enter to continue...")

                if direction == "left":
                    lane_idx = ego_lane_idx - 1
                elif direction == "right":
                    lane_idx = ego_lane_idx + 1
                elif direction == "ahead" or "behind":
                    lane_idx = ego_lane_idx
                dct_aligned["participant"][pariticipant]["lane_idx"] = lane_idx

                # ## Find the nearest vehicle in lane_idx
                if lane_idx != ego_lane_idx:
                    dct_vehicle[lane_idx][-1] = True
                else:
                    try:
                        # ## The vehicle is in the same lane as the ego vehicle
                        dct_vehicle[lane_idx][-2] = True
                    except:
                        pass
        idx_other_vehicle = len(dct_text["participant"]) - 1

        if self.debug:
            print("dct_vehicle:")
            pprint(dct_vehicle)

        for lane in dct_vehicle.keys():
            for idx, vehicle in enumerate(dct_vehicle[lane]):
                if not vehicle:
                    # ## The vehicle is not registered in the text modality
                    # ## Register the vehicle in the text modality
                    if lane < ego_lane_idx:
                        direction = "left"
                    elif lane > ego_lane_idx:
                        direction = "right"
                    else:
                        direction = "ahead"
                    idx_other_vehicle += 1

                    dct_text["participant"][f"other_vehicle_{idx_other_vehicle}"] = {
                        "lane_idx": lane,
                        "position_target": [f"{direction}", "ego_vehicle"],
                        # "speed_target": 0,
                        # "speed_limit": 0,
                        # "speed_limit_unit": "km/h"
                        "type": dct_visual[lane][idx][0],
                    }
                    dct_vehicle[lane][idx] = True

        if self.debug:
            print("dct_aligned:")
            pprint(dct_aligned)

        return dct_aligned

    def main(self):
        os.makedirs(self.save_dir, exist_ok=True)

        # ## Generate the source image name list
        ls_source_img_names = os.listdir(self.source_img_dir)
        ls_source_img_names = [x for x in ls_source_img_names if is_image_file(x)]
        ls_source_img_names.sort()

        # ## load the visual IR file name list
        ls_visual_ir_files = os.listdir(self.visual_ir_dir)
        ls_visual_ir_files = [x for x in ls_visual_ir_files if x.endswith(".yaml")]
        ls_visual_ir_files.sort()

        # ## load the textual IR file name list
        ls_textual_ir_files = os.listdir(self.textual_ir_dir)
        ls_textual_ir_files = [x for x in ls_textual_ir_files if x.endswith(".yaml")]
        ls_textual_ir_files.sort()

        for idx, source_img_name in enumerate(tqdm(ls_source_img_names)):
            source_img_name_no_ext = ".".join(source_img_name.split(".")[:-1])
            source_img_path = os.path.join(self.source_img_dir, source_img_name)

            # ## locate the visual IR file
            visual_ir_file_name = ls_visual_ir_files[idx]
            assert visual_ir_file_name.startswith(
                f"{source_img_name_no_ext}."
            ), f"{visual_ir_file_name} does not match {source_img_name}"

            # ## locate the textual IR file
            textual_ir_file_name = ls_textual_ir_files[idx]
            assert textual_ir_file_name.startswith(
                f"{source_img_name_no_ext}."
            ), f"{textual_ir_file_name} does not match {source_img_name}"

            merged_ir = self.align_two_modalities(
                textual_ir_fp=os.path.join(self.textual_ir_dir, textual_ir_file_name),
                visual_ir_fp=os.path.join(self.visual_ir_dir, visual_ir_file_name),
            )

            # ## Save results to YAML files
            with open(
                os.path.join(self.save_dir, f"{source_img_name_no_ext}.yaml"), "w"
            ) as f:
                yaml.dump(merged_ir, f)

            # # ## Save results to JSON files
            # with open(os.path.join(self.save_dir, f"{source_img_name}.json"), "w") as f:
            #     json.dump(merged_ir, f)


if __name__ == "__main__":
    runner = TrafficComposer(
        source_image_dir=SOURCE_IMAGE_DIR,
        textual_ir_dir=TEXTUAL_IR_LOAD_DIR,
        visual_ir_dir=VISUAL_IR_LOAD_DIR,
        save_dir=MERGED_IR_SAVE_DIR,
    )
    runner.main()
