# the notebook's main objective is to filter and prepare the dataset to train a summarizer on it.
import os, sys
from pathlib import Path
HOME = os.getcwd()

current = HOME 
while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
DATA_FOLDER = os.path.join(PARENT_DIR, 'src', 'data')
sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'data_analysis'))
sys.path.append(os.path.join(str(current), 'evaluation'))
sys.path.append(os.path.join(str(current), 'text_processing'))
sys.path.append(os.path.join(str(current), 'models'))


from datasets import load_dataset
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.training_utilities.pytorch_utilities import load_model
from train import prepare_labeled_data

CHECKPOINT = 't5-small'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def predict(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    # set the model and the tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(DEVICE)        
    model.load_state_dict(torch.load(checkpoint_path)) 


    # retrieve the test_split
    test_data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, 'test_split.csv'), split='train')
    test_data = test_data.map(lambda b: prepare_labeled_data(batch=b, tokenizer=tokenizer), 
                              batched=True).remove_columns(['source', 'target']) 


    with open(os.path.join(SCRIPT_DIR, "summarizer.txt"), 'w') as file:
        for i in range(len(test_data)):
            input_ids = test_data[i]['input_ids']
            attention_mask = test_data[i]['attention_mask']

            outputs = model.generate(
                input_ids=torch.tensor(input_ids).unsqueeze(0).to(DEVICE),
                attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(DEVICE),
                max_length=512,
                num_beams=2,
                early_stopping=True
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            file.write(f'{generated}\n')


if __name__ == '__main__':
    checkpoint_path = os.path.join(PARENT_DIR, 'checkpoints', 'best_summarizer', "11-5-6-33.pt")
    predict(checkpoint_path=checkpoint_path)
