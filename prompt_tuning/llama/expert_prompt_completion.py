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
    for i in range(len(promptsArray)):
        if len(promptsArray[i]) > 2000:
            promptsArray[i] = promptsArray[i][:2000]

    dialogs1: List[Dialog] = [
        [
            {
                "role": "system",
                "content": '''You are a chemical reaction data formatter. You will be provided unstructured data about the procedure for performing a chemical reaction and partially structured data about the molecules involved to convert into a single formatted JSON. I will give you the format of the required JSON and short descriptions of what is required in these fields. Here is the format of the JSON file, with a description of the required information:

 {{
    "reactants": [
        {{
            "name": "Reactant name",
            "amount": "Reactant amount",
            "mols": "Reactant moles", 
            "smiles": "Reactant SMILES"
        }},
        ...
    ],
    "spectators": [
        {{
            "name": "Spectator/solvent name",
            "amount": "Spectator/solvent amount",
            "mol": "Spectator/solvent moles", 
            "smiles": "Solvent SMILES"
        }},
        ...
    ],
    "products": [
        {{
            "name": "Product name",
            "amount": "Product amount",
            "mols": "Product moles", 
            "smiles": "Product SMILES"
        }},
        ...
    ],
    "yield": "reaction yield"
    "procedures": [
        {{
            "procedure": "Procedure type (Choose from preparation, reaction, workup, purification only. Workup refers to the addition of chemicals for quenching/neutralizing. Purification refers to subsequent isolation like filtration. Most reactions do contain both workup and purification, do not miss any of those, even within sentences. Make sure each sentence of the procedure is analysed for this)",
            "chemicals involved": "Chemicals involved in the reaction. For a workup or purification, only mention the chemicals used for the workup/purification",
            "description": "Description"
            "temperature": "temperature",
            "time": "time taken",
        }},
        ...
    ],
    "total_time": "Total amount of time for reaction"
    "product information": [
        {{
            "type": "Type of information (qualitative or quantitative or N/A only. There can be multiple parts of characterization informations separated by "; " or ". " characters. Separate these and include them one by one)",
            "descriptor": "Could be the color, physical state etc for qualitative, could be full details 1H NMR, 13C NMR, LCMS, MS or HPLC for quantitative, separated by ";" or ". ". Separate these and include them one by one"
        }},
        ...
}}

Few more guidelines:
Please interpolate between various parts of the given data, and feel free to use reasoning about chemicals, but do not add any extra information in this cleaned JSON. If any information is missing, use "N/A". If there seems to be an information mismatch between the structured and unstructured parts, inform accordingly. Most importantly, only output the formatted JSON and nothing else. Most importantly, only output the formatted JSON and nothing else.
'''
                ,
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
    fn = ckpt_dir.split('/')[0]  
    df.to_csv(f'chat_expert_{fn}.csv')

if __name__ == "__main__":
    fire.Fire(main)
