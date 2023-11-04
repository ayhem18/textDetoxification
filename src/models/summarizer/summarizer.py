"""This script contains the code to train The summarization model on a "Summary + Detoxification" portion of the collected data.
"""

import torch
import evaluate # a library used to compute standard metrics
import numpy as np

from typing import Iterable, Union, Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training_utilities.exp_tracking import create_summary_writer, report_results, _add_metric
from src.training_utilities.pytorch_utilities import save_model, cleanup


def compute_rouge(predictions: Iterable[str], 
                  references: Iterable[Union[str, Iterable[str]]],
                  rouge_type: str = 'rougeL',
                  ) -> Union[float, Dict]:
    # the main idea of this function is to compute the metrics on textual input
    # load a metric object
    rouge_obj = evaluate.load('rouge')
    results = rouge_obj.compute(predictions=predictions, references=references, use_stemmer=True)

    # make sure the 'rouge_type' belongs to the valid values
    valid_rouge_types = list(results.keys()) + ['all']
    if rouge_type not in valid_rouge_types:
        raise ValueError(f"The rouge metric is expected to belong to {valid_rouge_types}")
    
    if rouge_type == 'all':
        return results
    
    return results[rouge_type]


def train_per_epoch(summary_model: AutoModelForSeq2SeqLM, 
                    train_dataloader: DataLoader, 
                    optimizer: torch.optim,
                    scheduler: torch.optim.lr_scheduler = None,
                    device: str = None
                    ) -> float:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loss = 0    
    
    # set the model to the train mode
    summary_model.train()
    summary_model.to(device=device)

    for batch in train_dataloader:

        optimizer.zero_grad()
        # first put all the data into the device
        batch = {k: v.to(device) for k, v in batch.items()}
        # the next step is to pass the data to the model 
        model_output = summary_model(**batch)
        # extract the sequence to sequence loss
        sq2sq_loss = model_output.loss 
        train_loss += sq2sq_loss.item()

        # the backpropagation pass
        sq2sq_loss.backward()


        optimizer.step()

    if scheduler is not None:
        scheduler.step() 

    # make sure to divide the accumulated loss by the number of batches
    train_loss /= len(train_dataloader)

    return train_loss



def val_per_epoch(summary_model: AutoModelForSeq2SeqLM, 
                  val_dataloader: DataLoader ,
                  summary_tokenizer: AutoTokenizer = None, 
                  toxicity_loss_function: callable = None,
                  device: str = None, 
                  rouge_score: bool = True
                  ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_loss = 0    
    tox_loss = 0
    rouge_score = 0

    # set the model to the train mode
    summary_model.eval()
    summary_model.to(device=device)

    with torch.inference_mode():

        for batch in val_dataloader:
            # first put all the data into the device
            batch = {k: v.to(device) for k, v in batch.items()}
            # the next step is to pass the data to the model 
            model_output = summary_model(**batch)
            # extract the sequence to sequence loss
            sq2sq_loss = model_output.loss 
            val_loss += sq2sq_loss.item()
            
            # if the summary_tokenizer and the toxicity_loss_function is passed, compute the toxicity level 
            # of the generated sentences
            if summary_tokenizer is not None and toxicity_loss_function is not None:            
                batch_generate = {k: v for k, v in batch.items() if k != 'labels'}    
                output_decoded = summary_tokenizer.batch_decode(summary_model.generate(**batch_generate), skip_special_tokens=True)
                tox_loss += toxicity_loss_function(output_decoded, device=device)

            # compute the rouge metric depending on the value of 'compute_rouge' argument
            if rouge_score:
                # the compute_rouge function above expects textual input. The latter requires preparation
                if summary_tokenizer is None:
                    raise ValueError(f"Computing the ROUGE metric requires passing the summary_tokenizer")

                batch_generate = {k: v for k, v in batch.items() if k != 'labels'}    
                output_decoded = summary_tokenizer.batch_decode(summary_model.generate(**batch_generate), skip_special_tokens=True)

                # make sure to decode the labels as well
                labels = batch['labels'].cpu().numpy()
                # remove the extra padding 
                labels = np.where(labels != -100, labels, summary_tokenizer.pad_token_id)
                # decode the labels
                labels_decoded = summary_tokenizer.batch_decode(labels, skip_special_tokens=True)

                rouge_score += compute_rouge(labels_decoded)


    # make sure to divide the accumulated loss by the number of batches
    val_loss /= len(val_dataloader)
    tox_loss /= len(val_dataloader)
    rouge_score /= len(val_dataloader)

    losses = {"val_loss": val_loss}
    
    if summary_tokenizer is not None and toxicity_loss_function is not None:            
        losses['val_toxic_loss'] = tox_loss 

    if compute_rouge: 
        return losses, {"val_rouge_score": rouge_score}
    
    return  losses


def train_custom_summarizer(train_dataloader: DataLoader,
                            val_dataloader: DataLoader, 
                            
                            summary_model: AutoModelForSeq2SeqLM, 
                            summary_tokenizer: AutoTokenizer,
                            
                            toxicity_loss_function: callable, 
                            toxicity_coeff: float,
                            
                            optimizer: torch.optim, 
                            scheduler: torch.optim.lr_scheduler,

                            num_epochs: int = 5,
                            device: str = None,
                            log_dir: str = None, 
                            report_per_epoch: int = 5, 
                            ) -> Tuple[Dict[str, List[float]]]:
    
    # make sure to cleanup the GPU memory before starting to traing
    cleanup()
    
    # set the default device
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
    
    # set the SummaryWriter for visualization
    writer, save_path  = (None, None) if log_dir is None else (create_summary_writer(log_dir, return_path=True))

    # save the results somewhere
    train_losses = []
    val_losses = {"val_loss": [], "val_toxic_loss": []}

    best_train_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(1, num_epochs + 1)):
        # first the training part
        train_loss = train_per_epoch(train_dataloader=train_dataloader, 
                                summary_model=summary_model,
                                optimizer=optimizer, 
                                scheduler=scheduler, 
                                device=device
                                )
        
        val_results, val_metrics = val_per_epoch(val_dataloader=val_dataloader,
                                    summary_model=summary_model,
                                    summary_tokenizer=summary_tokenizer, 
                                    toxicity_loss_function=toxicity_loss_function, 
                                    device=device, 
                                    rouge_score=True
                                    )

        # make sure to save the best model
        if best_model is None:
            best_model = summary_model

        best_model = summary_model if train_loss < best_train_loss else best_model
        # update 'best_train_loss' 
        best_train_loss = min([best_train_loss, train_loss])

        train_losses.append(train_loss)
        for k, v in val_results.items():
            val_losses[k].append(v)            

        # add the train loss
        _add_metric(writer=writer, tag='train_loss', values=train_loss, epoch=epoch)
        _add_metric(writer=writer, tag='val_losses', values=val_results, epoch=epoch)
        _add_metric(writer=writer, tag='rouge_score', values=val_metrics, epoch=epoch)

        # report the results if needed  
        if epoch % report_per_epoch == 0: 
            report_results(train_losses_dict={"train_loss": train_loss}, val_losses_dict=val_results)

    # at the end save the model
    save_model(model=best_model, path=save_path)
    # return the results
    return train_losses, val_losses, best_model            
