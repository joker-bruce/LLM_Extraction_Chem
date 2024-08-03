from openai import OpenAI
import pandas as pd
import json

client = OpenAI()
output_list = []
df = pd.read_csv('/afs/crc.nd.edu/user/x/xhuang2/USPTO/Datasets /finetuning_data_1000_classes.csv')
df['input'] = df['molecules_reactants'] + ' ' + df['molecules_products'] + ' ' + df['text']
for i in range(len(df)):
    ##prompt1

    prompts = [
    {
    "role": "user",
    "content": df['input'][i]
}
]
    batchInstruction = {
    "role":
    "system",
    "content":
    """
    template = '''You are a chemical reaction data formatter. You will be provided unstructured data about the procedure for performing a chemical reaction and partially structured data about the molecules involved to convert into a single formatted JSON. I will give you the format of the required JSON and short descriptions of what is required in these fields. Here is the format of the JSON file, with a description of the required information:

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
    """
    #"Could you format the previous message into Json with the reactants:reactant amount, products:product amount, temperatures: temperature amount,  procedures, and total amount of time for reaction?"
}

    prompts.append(batchInstruction)

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=prompts, max_tokens=2048
)

    output_list.append(completion.choices[0].message.content)


df['extracted'] = output_list

df.to_csv('./gpt4_generated_uspto.csv')
