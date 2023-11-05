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


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer
from src.training_utilities.pytorch_utilities import cleanup


def prepare_sample(batch, tokenizer):
    # tokenize 'x'
    model_inputs = tokenizer(batch['source'], truncation=True)
    # tokenize 'y'  
    labels = tokenizer(text_target=batch["target"], truncation=True)
    # add it to the model's input
    model_inputs["labels"] = labels["input_ids"]
    # model_inputs["labels_attention_masks"] = labels['attention_mask']    
    return model_inputs

def load_the_data(tokenizer):
    train_data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, 'train_split.csv'), split='train', nrows=10)
    val_data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, 'val_split.csv'), split='train', nrows=10)
    test_data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, 'test_split.csv'), split='train', nrows=10)

    train_data = train_data.map(lambda b: prepare_sample(b, tokenizer=tokenizer), batched=True).remove_columns(['source', 'target'])
    val_data = val_data.map(lambda b: prepare_sample(b, tokenizer=tokenizer), batched=True).remove_columns(['source', 'target'])
    test_data = test_data.map(lambda b: prepare_sample(b, tokenizer=tokenizer), batched=True).remove_columns(['source', 'target'])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    return train_data, val_data, data_collator


from src.training_utilities.pytorch_utilities import get_module_device

class CustomTrainer(Trainer):
    def __init__(self, toxic_classifier, *args, **kwargs):
        self.toxic_classifier = toxic_classifier
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        model_output = model(**inputs)
        
        model_device = get_module_device(model)
        # extract the sequence to sequence loss
        s2s_loss = model_output.loss

        labels = inputs['labels']
        batch_size, max_sentence_length = labels.shape

        # reproduce the 
        prediction_ids = model.generate(inputs['input_ids'], max_length=max_sentence_length)

        attention_mask = torch.where(prediction_ids == self.tokenizer.pad_token_id,
                                     torch.zeros(*prediction_ids.shape).to(model_device), torch.ones(*prediction_ids.shape).to(model_device))
        
        toxic_output = self.toxic_classifier(input_ids=prediction_ids, attention_mask=attention_mask)
        toxic_loss = torch.mean(F.softmax(toxic_output.logits, dim=1)[:, 1])
        loss = s2s_loss + 0.1 * toxic_loss 
        return (loss, model_output) if return_outputs else loss 


def train(classifier_checkpoint: str = None):
    if classifier_checkpoint is None:
        model_checkpoint = os.path.join(PARENT_DIR, 'checkpoints', 's2s', 'toxic_classifier_checkpoints', 'checkpoint-600') 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the toxicity classifier = 
    toxic_classifier =  AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)

    # make sure the the toxic_classifier is indeed frozen
    for p in toxic_classifier.parameters():
        p.requires_grad = True

    # load the seq2seq model.
    checkpoint = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

    # get the data 
    train_data, val_data, data_collator = load_the_data(tokenizer=tokenizer)

    batch_size = 2
    num_epochs = 5
    learning_rate = 5e-5
    warmup_steps = 500
    weight_decay = 0.01

    sc_training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(PARENT_DIR, 'checkpoints', 's2s', 'seq2seq_checkpoints'),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=100,
        save_steps=1,
        eval_steps=10,
        overwrite_output_dir=True,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        report_to="none",
    )

    trainer = CustomTrainer(
        toxic_classifier=toxic_classifier,
        model=model,
        args=sc_training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator, 
        tokenizer=tokenizer 
    )

    cleanup()
    trainer.train() 


if __name__ == '__main__':
    train()
