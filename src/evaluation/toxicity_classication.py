import os
import gc
import tqdm
import torch
import numpy as np

from typing import List
from torch.nn.functional import softmax

from tqdm.auto import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# create a singletonInitializer to avoid repetitively loading the classification model 

_TOXIC_CLASSIFIER_CHECKPNT = 'SkolkovoInstitute/roberta_toxicity_classifier'


class EvalutionSingletonInitializer():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(EvalutionSingletonInitializer, cls).__new__(cls)            
            cls.instance.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cls.instance.toxic_tokenizer = AutoTokenizer.from_pretrained(_TOXIC_CLASSIFIER_CHECKPNT)
            cls.instance.toxic_classifier = AutoModelForSequenceClassification.from_pretrained(_TOXIC_CLASSIFIER_CHECKPNT)
        
        return cls.instance
    
    def get_device(self):
        return self.instance.device

    def get_toxic_tokenizer(self):
        return self.instance.toxic_tokenizer

    def get_toxic_classifier(self):
        return self.instance.toxic_classifier
    

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def toxic_classification(sentences: List[str], batch_size: int = None, return_logits: bool = False):
    # start by defining the singleton object
    singleton_obj = EvalutionSingletonInitializer()
    model, tokenizer = singleton_obj.get_toxic_classifier(), singleton_obj.get_toxic_tokenizer()
    
    device = singleton_obj.get_device()
    model.to(device=device)

    # a 'batch_size' implies classifying the entire data at once
    if batch_size is None:
        model_input = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        # make sure the input is in the same device as the model
        model_input = {k: v.to(device=device) for k, v in model_input.items()}
        # inference time    
        with torch.inference_mode():
            logits = model(**model_input).logits
            if return_logits:
                return logits            
            return softmax(logits, dim=1).cpu().numpy()[:, 1]


    results = []
    for i in trange(0, len(sentences), batch_size):
        # extract the batch
        batch_sentences = sentences[batch_size * i: (batch_size + 1) * i]
        # tokenize it
        model_input = tokenizer(batch_sentences, padding=True, truncation=True)
        with torch.inference_mode():
            logits = model(**model_input).logits        
            output = logits.tolist() if return_logits else softmax(logits, dim=1).cpu().numpy()[:, 1].tolist() 
            results.extend(output)
            # results.extend((logits.tolist()) if logits else (np.argmax(logits.cpu().numpy(), axis=1)[:, 1].tolist()))
    
    return results


def classify_preds(args, preds, soft=False):
    print('Calculating style of predictions')
    results = []

    model_name = args.classifier_path or 'SkolkovoInstitute/roberta_toxicity_classifier'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    for i in tqdm.tqdm(range(0, len(preds), args.batch_size)):
        batch = tokenizer(preds[i:i + args.batch_size], return_tensors='pt', padding=True)
        with torch.inference_mode():
            logits = model(**batch).logits
        if soft:
            result = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        else:
            result = (logits[:, 1] > args.threshold).cpu().numpy()
        results.extend([1 - item for item in result])
    return results


def detokenize(x):
    return x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )",")").replace("( ", "(")  # noqa





    