import torch


from torch import nn
from typing import Dict, Optional, Union
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from seq2seq_model import BertBasedEncoder, DecoderRNN

from src.training_utilities.exp_tracking import create_summary_writer, _add_metric, report_results

def train_per_epoch(encoder: BertBasedEncoder,
                    decoder: DecoderRNN, 
                    train_dataloader: DataLoader[torch.tensor],
                    loss_function: nn.Module,
                    e_opt: torch.optim, 
                    d_opt: torch.optim,
                    device: str=None,
                    ) -> Dict[str, float]:

    # set both components to 'train' mode
    encoder.train()
    decoder.train()

    # set both components to the right 'device'
    encoder.to(device)
    decoder.to(device)

    # set the train loss and metrics
    train_loss, train_acc = 0, 0

    for batch in train_dataloader:
        batch = {k:v.to(device) for k, v in batch.items()}

        batch_input = {k: v for k, v in batch.items() if k != 'labels' }

        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # first set both optimizers to zero gradients
        e_opt.zero_grad()
        d_opt.zero_grad()

        # extract the batch size, sequence length (with padding) 
        batch_size, seq_length = input_ids.shape

        _, hidden_state, cell_state = encoder.forward(batch_input)
        # pass the outputs of the encoder to the decoder
        decoder_outputs, _ , _ = decoder.forward(hidden_state, 
                                                 cell_state, 
                                                 max_seq_length=seq_length, 
                                                 batch_size=batch_size, 
                                                 target=labels)

        # the decoder's outputs are expected to be of shape (batch_size, seq_length, num_classes)
        loss = torch.zeros(size=()) # scalar vector
        
        for batch_index in range(batch_size):
            output_index = decoder_outputs[batch_index]
            y_index = labels[batch_index].squeeze().to(torch.long)

            seq_loss = torch.mean(loss_function(output_index, y_index))
            loss = torch.add(loss, seq_loss)
        
        # average the loss accross the batch
        loss /= batch_size 
        train_loss += loss.item()

        # take a backward step to calculate the gradients
        loss.backward()
        # optimize both for encoder and decoder
        e_opt.step()
        d_opt.step()

        y_pred = decoder_outputs.argmax(dim=-1)
        train_acc += (y_pred == labels).type(torch.float32).mean().item()

    train_acc = train_acc / len(train_dataloader)
    train_loss = train_loss / len(train_dataloader)

    return train_loss, train_acc


def val_per_epoch(encoder: BertBasedEncoder,
                  decoder: DecoderRNN,
                  dataloader: DataLoader[torch.tensor],
                  loss_function: nn.Module,
                  device:str=None,
                  ) -> Dict[str, float]:

    val_loss, val_acc = 0, 0
    # set both components to 'train' mode
    encoder.eval()
    decoder.eval()

    # set both components to the right 'device'
    encoder.to(device)
    decoder.to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (x, y) in enumerate(dataloader):
            # extract the batch size, sequence length (with padding) 
            batch_size, seq_length = x.shape

            _, hidden_state, cell_state = encoder.forward(x)
            # pass the outputs of the encoder to the decoder
            decoder_outputs, _ , _ = decoder.forward(hidden_state, cell_state, max_seq_length=seq_length, batch_size=batch_size)

            # the decoder's outputs are expected to be of shape (batch_size, seq_length, num_classes)
            loss = torch.zeros(size=()) # scalar vector
            
            for batch_index in range(batch_size):
                output_index = decoder_outputs[batch_index]
                y_index = y[batch_index].squeeze().to(torch.long)
                seq_loss = torch.mean(loss_function(output_index, y_index))
                loss = torch.add(loss, seq_loss)
            
            # average the loss accross the batch
            loss /= batch_size
            val_loss += loss.item()
            
            y_pred = decoder_outputs.argmax(dim=-1)
            val_acc += (y_pred == y).type(torch.float32).mean().item()

    # average by epoch
    val_acc = val_acc / len(dataloader)
    val_loss = val_loss / len(dataloader)

    return val_loss, val_acc


def strain_model(
                encoder: BertBasedEncoder, 
                decoder: DecoderRNN,
                train_dataloader: DataLoader[torch.Tensor],
                test_dataloader: DataLoader[torch.Tensor],
                loss_function,
                e_opt, 
                d_opt,
                epochs: int = 5,
                log_dir: Optional[Union[Path, str]] = None,
                save_path: Optional[Union[Path, str]] = None,
                ):

    save_path = save_path if save_path is not None else log_dir

    performance_dict = {'train_loss': [],
                        'val_loss': [],
                        'train_accuracy': [],
                        'val_accuracy': []}
    

    # best_model, best_loss = None, None
    min_training_loss, no_improve_counter, best_model = float('inf'), 0, None

    # before proceeding with the training, let's set the summary writer
    writer = None if log_dir is None else create_summary_writer(log_dir)

    for _ in tqdm(range(epochs)):
        epoch_train_loss, epoch_train_acc = train_per_epoch(encoder=encoder,
                                                            decoder=decoder,
                                                            e_opt=e_opt,
                                                            d_opt=d_opt,
                                                            train_dataloader=train_dataloader,
                                                            loss_function=loss_function)

        epoch_val_loss, epoch_val_acc = val_per_epoch(encoder=encoder,
                                                      decoder=decoder,
                                                      dataloader=test_dataloader,
                                                      loss_function=loss_function)

        no_improve_counter = no_improve_counter + 1 if min_training_loss < epoch_train_loss else 0

        if min_training_loss > epoch_train_loss:
            # save the model with the lowest training error
            min_training_loss = epoch_train_loss

        report_results(train_losses_dict={"train_loss": epoch_train_loss, "train_acc": epoch_train_acc}, 
                        val_losses_dict={"val_accuracy": epoch_val_acc, "val_loss": epoch_val_loss})

        _add_metric(writer,
                    tag='loss', 
                    values={"train_loss": epoch_train_loss, "val_loss": epoch_val_loss},
                    epoch=_)

        _add_metric(writer,
                    tag='accuracy', 
                    values={"train_loss": epoch_train_acc, "val_loss": epoch_val_acc},
                    epoch=_)
        


    return performance_dict
