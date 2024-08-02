# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import math
from llama import Llama, Dialog
import pandas as pd
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    df = pd.read_csv('/afs/crc.nd.edu/user/x/xhuang2/llama_test/llama-recipes/recipes/finetuning/datasets/combined_finetuning_data_test.csv')
    promptsArray = df['input'].tolist()
    dialogs1: List[Dialog] = [
        [
            {
                "role": "system",
                "content": """
                        Could you format the previous message into Json with the reactants:reactant amount, products:product amount, temperatures: temperature amount,  procedures, and total amount of time for reaction?
                        """,
            },
            {"role": "user", "content": f"""
            {j}
            """},
        ] for j in promptsArray
    ]
    temp = []
    total_len = len(dialogs1)
    for i in range(math.ceil(total_len/max_batch_size)):
        
        dialogs = dialogs1[i*max_batch_size:(i+1)*max_batch_size]
        if(i == (math.ceil(total_len/max_batch_size) - 1)):
            dialogs = dialogs1[i*max_batch_size:total_len]
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        for result in results:
            temp.append({result['generation']['content']})
    df['extracted'] = temp
    df.to_csv(f'chat_naive_new_{ckpt_dir}.csv')

if __name__ == "__main__":
    fire.Fire(main)
