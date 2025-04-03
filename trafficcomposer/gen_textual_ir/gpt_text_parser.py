import openai
from openai import OpenAI
import time
import os
from tqdm import tqdm
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
print(
    f"Added Project Root to sys.path: {os.path.join(os.path.dirname(__file__), '../..')}"
)

from trafficcomposer.gen_textual_ir.gen_textual_ir import gen_textual_ir
from trafficcomposer.gen_textual_ir.config_textual import (
    DESCRIPTION_DIR,
    TEXTUAL_IR_SAVE_DIR,
    GPT4O_MINI_TEXTUAL_IR_SAVE_DIR,
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIClientRunner:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI()

    def call_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            # max_tokens=300,
        )
        # max_tokens: Specifies the maximum number of tokens to generate in the response. If the response would be longer than this number of tokens, it will be truncated.
        return response.choices[0].message.content

    def try_call_gpt(self, prompt, max_tries=3):
        for _ in range(max_tries):
            try:
                return self.call_gpt(prompt)
            except Exception as e:
                print(e)
                time.sleep(5)
        return None

    def __call__(self, prompt):
        return self.try_call_gpt(prompt, max_tries=3)


if __name__ == "__main__":
    model = "gpt-4o"
    save_dir = TEXTUAL_IR_SAVE_DIR

    # model = "gpt-4o-mini"
    # save_dir = GPT4O_MINI_TEXTUAL_IR_SAVE_DIR

    llm_runner = OpenAIClientRunner(model=model)
    gen_textual_ir(
        dir_description=DESCRIPTION_DIR,
        save_dir=save_dir,
        llm_runner=llm_runner,
        is_continue=False,  # Set to True to ignore existing files in the save directory.
        debug=True,
    )
