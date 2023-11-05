# the notebook's main objective is to filter and prepare the dataset to train a summarizer on it.
import random
import torch
import transformers
random.seed(69)
torch.manual_seed(69)
torch.cuda.manual_seed(69)
transformers.set_seed(69)

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

import pandas as pd
from torch import nn
from transformers import Trainer
from transformers import TrainingArguments
from src.training_utilities.pytorch_utilities import cleanup
from typing import Dict
from torch.nn.functional import softmax

import datasets
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, BartForSequenceClassification, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModel, AutoTokenizer

def prepare_toxicity_data():
    data = pd.read_csv(os.path.join(DATA_FOLDER, 'toxic_train.csv'), usecols= lambda x: x !='id')

    # let's convert all the different sub toxicity-labels into a single label.
    data['is_toxic'] = ((data['toxic'] + data['severe_toxic'] + data['obscene'] + data['threat'] + data['insult'] + data['identity_hate']) > 0).astype(int)
    
    new_data= data.drop(columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']).rename(columns={'comment_text': 'text'})
    
    # new_data.to_csv(os.path.join(DATA_FOLDER, 'toxicity_data.csv'), index=False) 

    toxic, non_toxic = new_data[new_data['is_toxic'] == 1], new_data[new_data['is_toxic'] == 0]
    # let's make a final balanced dataset
    num_samples = len(toxic) 
    balanced_dataset = pd.concat([toxic, non_toxic.iloc[:num_samples, :]])

    # save the balanced dataset
    balanced_dataset.to_csv(os.path.join(DATA_FOLDER, 'toxicity_data.csv'), index=False)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# load tokenizer and model weights
toxic_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
toxic_classifier = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier') 


def process_labels(batch: Dict, device: str):
    model_input = toxic_tokenizer(batch['text'], return_tensors='pt', truncation=True, padding=True)  
    model_input = {k: v.to(device) for k, v in model_input.items()}
    toxic_classifier.to(device)
    model_input['label'] = softmax(toxic_classifier(**model_input).logits, dim=1)
    return model_input

def process_data(batch: Dict, tokenizer):
    model_input = tokenizer(batch['text'], truncation=True)
    model_input['label'] = batch['label']
    return model_input


def set_model():
    checkpoint = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # freeze the entire model but the classification head
    for n, p in model.named_parameters():
        if n not in ["classification_head.out_proj.bias", 
                    'classification_head.dense.weight', 
                    'classification_head.dense.bias', 
                    'classification_head.out_proj.weight']:
            
            p.requires_grad = False
        else:
            print(n)

    return model, tokenizer


from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import src.data_preparation.prepare_data as pdr


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss 
    
def train_classifier():
    model, tokenizer = set_model()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, 'toxicity_data.csv'), split='train')

    d = data.map(lambda b : process_labels(b, device=DEVICE), batched=True, batch_size=4)
    d = d.map(lambda b : process_data(b, tokenizer=tokenizer), batched=True, batch_size=32).remove_columns(['is_toxic', 'text'])

    train_data, val_data, test_data = pdr.data_split(all_data=d, train_portion=0.9, val_portion=0.05)

    learning_rate = 5e-5
    training_args = TrainingArguments(os.path.join(PARENT_DIR, 'checkpoints', 's2s', 'toxic_classifier_checkpoints'),
                                    per_device_train_batch_size=32, 
                                    per_device_eval_batch_size=32, 
                                    num_train_epochs=3, 
                                    warmup_steps=500, 
                                    weight_decay=0.001, 
                                    learning_rate=learning_rate, 
                                    logging_steps=100, 
                                    save_steps=100, 
                                    report_to='none'
                                    )

    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    cleanup()
    trainer.train()


if __name__ == '__main__':
    train_classifier()