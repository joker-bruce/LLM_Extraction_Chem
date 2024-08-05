## GPT Finetuning
### Setup
```
python -m venv openai-venv
pip install -r requirements.txt
```
You'll need to get the open API key from [Openai](https://openai.com/index/openai-api/).  
Then perform the following:
```
source openai-venv/bin/activate
export OPENAI_API_KEY=your/api/key/here
python openai-test.py --chem_file path/to/your/file --model_name your/chosen/openai/model
```
After running through the code, the output would be saved as output.csv.

