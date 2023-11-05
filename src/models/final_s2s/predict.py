# the notebook's main objective is to filter and prepare the dataset to train a summarizer on it.
import os, sys
from pathlib import Path
HOME = os.getcwd()

current = HOME 
while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
DATA_FOLDER = os.path.join(PARENT_DIR, 'src','data')
data_path = os.path.join(DATA_FOLDER, 'filtered.tsv')

sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'data_analysis'))
sys.path.append(os.path.join(str(current), 'evaluation'))
sys.path.append(os.path.join(str(current), 'text_processing')) 


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datasets import load_dataset

from train_custom_seq2seq import prepare_sample

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def predict():
    checkpoint = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    seq2seq_checkpoint = os.path.join(PARENT_DIR, 
                                      'checkpoints', 
                                      's2s', 
                                      'seq2seq_checkpoints', 
                                      'checkpoint-21200')
    model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_checkpoint).to('cuda')

    test_data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, 'test_split.csv'), split='train')
    test_data = test_data.map(lambda b: prepare_sample(b, tokenizer=tokenizer), batched=True).remove_columns(['source', 'target'])

    with open(os.path.join(SCRIPT_DIR, "custom_s2s.txt"), 'w') as file:
        for i in range(len(test_data)):
            input_ids = test_data[i]['input_ids']
            attention_mask = test_data[i]['attention_mask']

            outputs = model.generate(
                input_ids=torch.tensor(input_ids).unsqueeze(0).to('cuda'),
                attention_mask=torch.tensor(attention_mask).unsqueeze(0).to('cuda'),
                max_length=512,
                num_beams=2,
                early_stopping=True
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            file.write(f'{generated}\n')


if __name__ == '__main__':
    predict()