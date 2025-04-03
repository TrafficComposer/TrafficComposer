# import requests

import openai
from openai import OpenAI

import datetime
import json
from tqdm import tqdm

import time


# Load api key from .env file
import os
import dotenv

dotenv.load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
print(
    f"Added Project Root to sys.path: {os.path.join(os.path.dirname(__file__), '../../..')}"
)

from trafficcomposer.baseline.multi_modal_gpt.config_baseline_gpt4o import (
    SOURCE_IMAGE_DIR,
    DESCRIPTION_DIR,
    BASELINE_GPT4O_SAVE_PATH,
)

from trafficcomposer.gen_visual_ir.utils import is_image_file
from trafficcomposer.gen_textual_ir.gpt_text_parser import OpenAIClientRunner
from trafficcomposer.gen_textual_ir.gen_textual_ir import post_process
from multi_modal_gen_prompt import gen_multi_modal_prompt


def main(
    source_image_dir=SOURCE_IMAGE_DIR,
    description_dir=DESCRIPTION_DIR,
    save_dir=BASELINE_GPT4O_SAVE_PATH,
    resume=False,
    gpt_model="gpt-4o",
    debug=False,
):

    # ## Build save directory
    if os.path.exists(save_dir):
        while True:
            dir_operation = input(
                f"WARNING: directory {save_dir} already exists! \nPress Enter to continue; \nInput `delete` to delete the original dir and create a new one; \nPress Ctrl-c or Input `exit` to exit; \nYour input: "
            )
            if dir_operation.lower() == "delete":
                print("Deleting the original directory and creating a new one...")
                os.rmdir(save_dir)
                os.makedirs(save_dir)
                break
            elif dir_operation.lower() == "exit":
                print("Exiting...")
                return
            elif dir_operation.lower() in ("continue", ""):
                print("Continuing without modifying the original directory...")
                break
    else:
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir)

    # ## Generate the source image name list
    ls_source_img_names = os.listdir(source_image_dir)
    ls_source_img_names = [x for x in ls_source_img_names if is_image_file(x)]
    ls_source_img_names.sort()

    # ## load the description file name list
    ls_description_files = os.listdir(description_dir)
    ls_description_files = [x for x in ls_description_files if x.endswith(".txt")]
    ls_description_files.sort()

    gpt_runner = OpenAIClientRunner(model=gpt_model)

    for idx, source_img_name in enumerate(tqdm(ls_source_img_names)):
        source_img_name_no_ext = ".".join(source_img_name.split(".")[:-1])
        source_img_path = os.path.join(source_image_dir, source_img_name)
        assert os.path.exists(source_img_path), f"{source_img_path} does not exist"

        # ## locate the traffic description file
        description_file_name = ls_description_files[idx]
        assert description_file_name.startswith(
            f"{source_img_name_no_ext}."
        ), f"{description_file_name} does not match {source_img_name}"

        desc_file_path = os.path.join(description_dir, description_file_name)
        assert os.path.join(description_dir, desc_file_path), f"{desc_file_path} does not exist"
        with open(desc_file_path, "r") as scenario_file:
            traffic_scenario_description = scenario_file.readlines()
        traffic_scenario_description = "".join(traffic_scenario_description)

        if traffic_scenario_description == "":
            print(f"file {desc_file_path} is empty")
            continue

        save_file_name = os.path.basename(desc_file_path)
        if save_file_name.endswith(".txt"):
            save_file_name = save_file_name[:-4] + ".yaml"
        save_path = os.path.join(save_dir, save_file_name)
        if resume and os.path.exists(save_path):
            print(f"Skipping {source_img_name} as {save_path} already exists.")
            continue

        if debug:
            print(f"========== Input ==========")
            print(f"source_img_path: {source_img_path}")
            print(f"desc_file_path: {desc_file_path}")
            print(traffic_scenario_description)
            print()

        prompt = gen_multi_modal_prompt(
            traffic_description=traffic_scenario_description,
            image_path=source_img_path,
        )
        baseline_ir = gpt_runner(prompt)
        time.sleep(1)

        baseline_ir = post_process(baseline_ir)

        print(f"========== Output ==========")
        print(baseline_ir)
        print()

        with open(save_path, "w") as fp:
            # json.dump(scenario_info, fp)
            fp.write(baseline_ir)

        # # ## Save results to YAML files
        # with open(os.path.join(save_dir, f"{source_img_name}.yaml"), "w") as f:
        #     yaml.dump(baseline_ir, f)

        # # ## Save results to JSON files
        # with open(os.path.join(self.save_dir, f"{source_img_name}.json"), "w") as f:
        #     json.dump(merged_ir, f)


if __name__ == "__main__":
    # gpt_model = "gpt-4o-mini"
    gpt_model = "gpt-4o"
    main(
        source_image_dir=SOURCE_IMAGE_DIR,
        description_dir=DESCRIPTION_DIR,
        save_dir=BASELINE_GPT4O_SAVE_PATH,
        gpt_model=gpt_model,
        resume=True,    # Set to True to ignore existing files in the save directory.
        debug=True,
    )
