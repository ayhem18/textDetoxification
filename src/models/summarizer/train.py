# the notebook's main objective is to filter and prepare the dataset to train a summarizer on it.
import os, sys
from pathlib import Path
from datasets import load_dataset

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# create a DataCollator for padding for the sequence to sequence models
from transformers import DataCollatorForSeq2Seq
# let's write a function to compute the summarization + toxicity loss
from torch.nn.functional import softmax
from typing import Union
from torch.utils.data import DataLoader


CHECKPOINT = 't5-small'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HOME = os.getcwd()
current = HOME 
while 'src' not in os.listdir(current):
    current = Path(current).parent


PARENT_DIR = str(current)
DATA_FOLDER = os.path.join(PARENT_DIR, 'data')
data_path = os.path.join(DATA_FOLDER, 'filtered.tsv')
sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'data_analysis'))
sys.path.append(os.path.join(str(current), 'evaluation'))
sys.path.append(os.path.join(str(current), 'text_processing'))
sys.path.append(os.path.join(str(current), 'models')) 

from src.evaluation.toxicity_classication import EvalutionSingletonInitializer
from src.evaluation import toxicity_classication as tc
import src.data_preparation.prepare_data as pdr


# the prefix to add to the textual input
TASK_PREFIX = 'summarize: '
def prepare_labeled_data(batch, tokenizer):
    # add the task predix to each sentence
    inputs = [TASK_PREFIX + doc for doc in batch["source"]]
    # tokenize 'x'
    model_inputs = tokenizer(inputs, truncation=True, max_length=1028)
    # tokenize 'y'  
    labels = tokenizer(text_target=batch["target"], truncation=True)
    # add it to the model's input
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_data(tokenizer):
    summary_data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, 'summarized_data.csv'), split='train')
    # split the data into train, val and test
    train_data, val_data, _ = pdr.data_split(all_data=summary_data)

    # apply the prepare_labeled_data function
    train_data = train_data.map(lambda b: prepare_labeled_data(b, tokenizer=tokenizer),
                                batched=True).remove_columns(['source', 'target'])
    
    val_data = val_data.map(lambda b: prepare_labeled_data(b, tokenizer=tokenizer), 
                            batched=True).remove_columns(['source', 'target'])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=CHECKPOINT)
    # we are now ready to create the dataloader
    train_dl = DataLoader(dataset=train_data, batch_size=4, shuffle=True, collate_fn=data_collator)
    val_dl = DataLoader(dataset=val_data, batch_size=4, shuffle=False, collate_fn=data_collator)

    return train_data, val_data, train_dl, val_dl


def set_the_model():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(DEVICE)
    return model, tokenizer



def toxic_summary_model_loss(output_decoded: torch.Tensor, 
                             device,
                             return_tensor: bool=False) -> Union[float, torch.Tensor]:
    
    singleton_obj = EvalutionSingletonInitializer()
    tc_tokenizer, tc_classifier = singleton_obj.get_toxic_tokenizer(), singleton_obj.get_toxic_classifier()

    # make sure to freeze their parameters
    for p in tc_classifier.parameters():
        p.requires_grad = False

    tc_classifier.to(device)
    # tokenize
    model_input = tc_tokenizer(output_decoded, return_tensors='pt', padding=True, truncation=True)
    # set the input to the device
    model_input = {k: v.to(device) for k, v in model_input.items()}
    # pass through the model
    output = tc_classifier(**model_input)
    
    loss = torch.mean(softmax(output.logits, dim=1)[:, 1])
    
    if return_tensor: 
        loss.requires_grad=True
        return loss
    
    return loss.item()


from src.models.summarizer import summarizer  as ss
import src.training_utilities.exp_tracking as et
# let's define some of the training parameters
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR


def train():
    # first set the model
    model, tokenizer = set_the_model()

    # get the data
    train_data, val_data, train_dl, val_dl = prepare_data(tokenizer)

    optimizer = Adam(model.parameters(), lr=2 * 10 ** -5)
    scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.5,total_iters=100)

    _, _, best_model = ss.train_custom_summarizer(train_dataloader=train_dl, 
                                            val_dataloader=val_dl,
                                            summary_model=model,
                                            summary_tokenizer=tokenizer, 
                                            toxicity_loss_function=toxic_summary_model_loss,
                                            toxicity_coeff=0.5, 
                                            optimizer=optimizer, 
                                            scheduler=scheduler, 
                                            num_epochs=20,   
                                            report_per_epoch=1,
                                            log_dir=os.path.join(HOME, 'runs'),
                                            save_path=os.path.join(PARENT_DIR, 'checkpoints', 'best_summarizer')
                                            )

    return best_model


if __name__ == '__main__':
    train()
