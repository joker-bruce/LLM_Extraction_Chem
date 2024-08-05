## Instruction for prompt_tuning llama
### Setup environment
```
python -m venv pt_venv
pip install -r requirements.txt
```
Go to [Meta Llama download link](https://llama.meta.com/llama-downloads/) to download models and place them into [models](https://github.com/joker-bruce/LLM_Extraction_Chem/tree/main/models) folder.  

### Run the code
we run the code on 4 A100 cards. For Llama2-7b-chat, we run the inference with only 1 A100 card. With Llama2-13b-chat and Llama2-70b-chat, we run with 2 cards and 4 cards respectively. 
```
#example with 13b model
source pt-venv/bin/activate
torchrun --nproc_per_node 2 prompt_tuning_llama.py --prompt_type no_prompt --chem_file location-of-the-reaction-info-file --ckpt_dir model_directory --tokenizer_path tokenizer.model --max_seq_len 4096
```
There will also be an output csv file with output as a new column to the original csv file.
