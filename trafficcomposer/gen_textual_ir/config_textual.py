import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
print(
    f"Added Project Root to sys.path: {os.path.join(os.path.dirname(__file__), '../..')}"
)

from trafficcomposer.config import (
    DATA_ROOT_DIR,
    LLAMA3_1_HOME_PATH,
    LLAMA3_HOME_PATH,
    LLAMA2_HOME_PATH,
)

DATA_ROOT_DIR = DATA_ROOT_DIR
LLAMA3_1_HOME_PATH = LLAMA3_1_HOME_PATH
LLAMA3_HOME_PATH = LLAMA3_HOME_PATH
LLAMA2_HOME_PATH = LLAMA2_HOME_PATH


DESCRIPTION_DIR = os.path.join(DATA_ROOT_DIR, "inputs/textual_description")

TEXTUAL_IR_SAVE_DIR = os.path.join(DATA_ROOT_DIR, "results/textual/gpt-4o")
GPT4O_MINI_TEXTUAL_IR_SAVE_DIR = os.path.join(DATA_ROOT_DIR, "results/textual/gpt-4o-mini")
TEXTUAL_IR_SAVE_ROOT_DIR = os.path.join(DATA_ROOT_DIR, "results/textual")


LLAMA3_1_70B_INSTRUCT_TEXTUAL_IR_SAVE_DIR = os.path.join(
    TEXTUAL_IR_SAVE_ROOT_DIR, "llama-3.1-70B-Instruct"
)
LLAMA3_1_8B_INSTRUCT_TEXTUAL_IR_SAVE_DIR = os.path.join(
    TEXTUAL_IR_SAVE_ROOT_DIR, "llama-3.1-8B-Instruct"
)


LLAMA3_70B_INSTRUCT_TEXTUAL_IR_SAVE_DIR = os.path.join(
    TEXTUAL_IR_SAVE_ROOT_DIR, "llama-3-70B-Instruct"
)
LLAMA3_8B_INSTRUCT_TEXTUAL_IR_SAVE_DIR = os.path.join(
    TEXTUAL_IR_SAVE_ROOT_DIR, "llama-3-8B-Instruct"
)

LLAMA2_70B_CHAT_TEXTUAL_IR_SAVE_DIR = os.path.join(
    TEXTUAL_IR_SAVE_ROOT_DIR, "llama-2-70B-Chat"
)
LLAMA2_13B_CHAT_TEXTUAL_IR_SAVE_DIR = os.path.join(
    TEXTUAL_IR_SAVE_ROOT_DIR, "llama-2-13B-Chat"
)
LLAMA2_7B_CHAT_TEXTUAL_IR_SAVE_DIR = os.path.join(
    TEXTUAL_IR_SAVE_ROOT_DIR, "llama-2-7B-Chat"
)
