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
data = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'), usecols= lambda x: x !='id')
data.head()

# let's convert all the different sub toxicity-labels into a single label.
data['is_toxic'] = ((data['toxic'] + data['severe_toxic'] + data['obscene'] + data['threat'] + data['insult'] + data['identity_hate']) > 0).astype(int)
def prepare_data(row):
    row['is_toxic'] = int(row['toxic'] + row['severe_toxic'] + row['obcene'] + row['threat'] + row['insult'] + row['identity_hate'] > 0)
    return row 
# new_data = data.apply(prepare_data, axis='index')
new_data= data.drop(columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']).rename(columns={'comment_text': 'text'})
new_data['is_toxic'].value_counts()
# new_data.to_csv(os.path.join(DATA_FOLDER, 'toxicity_data.csv'), index=False) 


toxic, non_toxic = new_data[new_data['is_toxic'] == 1], new_data[new_data['is_toxic'] == 0]
# let's make a final balanced dataset
num_samples = len(toxic) 
balanced_dataset = pd.concat([toxic, non_toxic.iloc[:num_samples, :]])


# save the balanced dataset
balanced_dataset.to_csv(os.path.join(DATA_FOLDER, 'toxicity_data.csv'), index=False)

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModel, AutoTokenizer
# load tokenizer and model weights
toxic_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
toxic_classifier = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier') 

import datasets
data = datasets.load_dataset('csv', data_files=os.path.join(DATA_FOLDER, 'toxicity_data.csv'), split='train')

import torch
from transformers import AutoTokenizer, BartForSequenceClassification, AutoModelForSequenceClassification

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


from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# d.save_to_disk(os.path.join(DATA_FOLDER, f'toxicity_data'))
d  = datasets.load_from_disk(os.path.join(DATA_FOLDER, f'toxicity_data')) 

import src.data_preparation.prepare_data as pdr
train_data, val_data, test_data = pdr.data_split(all_data=d, train_portion=0.9, val_portion=0.05)

from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss 
    

from transformers import TrainingArguments

batch_size = 64
num_epochs = 1
learning_rate = 5e-5
warmup_steps = 500
weight_decay = 0.01


training_args = TrainingArguments(os.path.join(os.getcwd(), "toxic_classifier_checkpoints"), 
                                  per_device_train_batch_size=128, 
                                  per_device_eval_batch_size=128, 
                                  num_train_epochs=3, 
                                  warmup_steps=500, 
                                  weight_decay=0.001, 
                                  learning_rate=learning_rate, 
                                  logging_steps=100
                                  )

trainer = CustomTrainer(
    model,
    training_args,
    train_dataset=d,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


from src.training_utilities.pytorch_utilities import cleanup
cleanup()
trainer.train()