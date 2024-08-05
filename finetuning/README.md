## Setup
Please enter the llama-recipe and follow the set up instruction for new environent for finetuning the llama model.

## Download the models
Please go the huggingface to download [Llama](https://huggingface.co/meta-llama) as you wish. We use one A100 GPU to finetune [llama-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) and two A100 GPUs for [llama-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf) with PEFT(LoRA) for 200 epochs. You can use the PEFT model for 7b(7b_PEFT_model) and 13b(13b_PEFT_model) for evaluation purposes.

## Run Code
Activate the virtual environment and go into [llama-recipe](https://github.com/joker-bruce/llama-recipes/tree/main/recipes/quickstart/finetuning). 
Run the following code:
```
torchrun --nnodes 1 --nproc_per_node 2  finetuning.py --num_epoch 200 --enable_fsdp --dataset uspto_dataset --model_name path/to/huggingface/model --use_peft --peft_method lora --output_dir location/to/store/PEFT/model
```
