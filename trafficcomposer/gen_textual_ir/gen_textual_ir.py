import os
from tqdm import tqdm
import time
import re
import yaml


from trafficcomposer.gen_textual_ir.text_parser_gen_prompt import gen_prompt


def post_process(content, start_keyword="<YAML>", end_keyword="</YAML>"):
    """
    Extracts content from a string that is between the keywords start_keyword and end_keyword.
    Args:
        content (str): The string to extract content from.
        start_keyword (str): The keyword that marks the beginning of the content.
        end_keyword (str): The keyword that marks the end of the content.
    """
    pattern = f"{start_keyword}(.*?){end_keyword}"

    match = re.search(pattern, content, re.DOTALL)

    if match:
        ans = match.group(1).strip()

        # Load to YAML for position fix
        try:
            yaml_data = yaml.safe_load(ans)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return None

        # Initialize a dictionary to keep track of updated entries
        updated_entries = {}

        # Step 1: Process the YAML data to create updated_entries
        for entry_name, entry in yaml_data.get('participant', {}).items():
            position_target = entry.get('position_target')
            position_relation = entry.get('position_relation')

            if position_target == 'ego vehicle':
                updated_entries[entry_name] = position_relation

        # Initialize a dictionary to keep track of operation counts
        operation_lst = {}

        # Step 2: Update entries based on updated_entries
        for entry_name, entry in yaml_data.get('participant', {}).items():
            position_target = entry.get('position_target')
            operation_lst[position_target] = operation_lst.get(position_target, 0) + 1

            if position_target in updated_entries:
                current_relation = entry.get('position_relation')
                new_relation = updated_entries[position_target]

                if current_relation != new_relation:
                    entry['position_relation'] = f"{current_relation} {new_relation}"
                entry['position_target'] = 'ego vehicle'
            else:
                entry['position_relation'] = "None"

        ans = yaml.dump(yaml_data, default_flow_style=False)

        return ans
    else:
        return None


def gen_textual_ir(
    dir_description, save_dir, llm_runner, is_continue=False, debug=False
):
    """
    The main function to generate Te
    Args:
        dir_description (str): The path of the folder containing the scenario description files (.txt).
    """

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

    # ## Alarm if running in is_continue mode
    if is_continue:
        input(
            "WARNING: Running in continue mode! This will NOT update the existing files in the save directory! Press ENTER to continue; Press Ctrl-c to exit."
        )

    ls_description_txt = os.listdir(dir_description)
    ls_description_txt = [i for i in ls_description_txt if i.endswith(".txt")]
    ls_description_txt.sort()

    for desc_file_name in tqdm(ls_description_txt):
        desc_file_path = os.path.join(dir_description, desc_file_name)

        save_file_name = os.path.basename(desc_file_path)
        if save_file_name.endswith(".txt"):
            save_file_name = save_file_name[:-4] + ".yaml"
        save_path = os.path.join(save_dir, save_file_name)
        if is_continue and os.path.exists(save_path):
            print(f"WARNING: File {save_path} already exists! Skipping...")
            continue

        with open(desc_file_path, "r") as scenario_file:
            traffic_scenario_description = scenario_file.readlines()
        traffic_scenario_description = "".join(traffic_scenario_description)
        if traffic_scenario_description == "":
            print(f"WARNING: file {desc_file_path} is empty! Skipping...")
            continue

        if debug:
            print(f"===== File: {desc_file_path} =====")
            print(f"===== Input: =====")
            print(traffic_scenario_description)
            print(f"-- End of Input --")

        prompt = gen_prompt(traffic_scenario_description)
        textual_ir = llm_runner(prompt)
        if debug:
            print(f"===== Raw Output of GPT: =====")
            print(textual_ir)
            print(f"-- End of Raw Output of GPT --")
        time.sleep(1)

        textual_ir = post_process(textual_ir)

        print(f"Output:")
        print(textual_ir)
        print()

        with open(save_path, "w") as fp:
            # json.dump(scenario_info, fp)
            fp.write(textual_ir)

        if debug:
            print(f"Saved to {save_path}")
