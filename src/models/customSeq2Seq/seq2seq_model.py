"""_summary_
"""

import torch
from torch import nn

from torch.utils.data import DataLoader
from typing import Optional, Dict
from transformers import AutoModel, AutoTokenizer


NOTEBOOK_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = 'distilbert-base-uncased'  # let's keep it simple as for the first iteration
MODEL = AutoModel.from_pretrained(CHECKPOINT).to(NOTEBOOK_DEVICE)
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)


class BertBasedEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int = None,
                 num_layers: int = 1,
                 dropout: float = 0.3,
                 checkpoint: str = 'distilbert-base-uncased',
                 freeze: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # first make sure to initialize the bert model
        self.bert_model = AutoModel.from_pretrained(checkpoint)
        
        # make sure to freeze bert
        if freeze: 
            for p in self.bert_model.parameters():
                p.requires_grad=False
        
        # extract the embedding dimensions and the vocabulary size
        emb_dim = self.bert_model.config.hidden_size

        # set the default value for the hidden
        self.hidden_dim = emb_dim // 2 if hidden_dim is None else hidden_dim
        # set the dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # set the RNN
        self.rnn = nn.LSTM(input_size=emb_dim,
                           hidden_size=self.hidden_dim,
                           dropout=dropout,
                           num_layers=num_layers,
                           bidirectional=True,  # bidirectional RNNs are more powerful
                           batch_first=True,   # easier manipulation
                           )

        # 2: comes from the fact that the lstm is bidirectional, the rest is similar to the LSTM documentation Pytorch
        self.hidden_state_dim = 2 * num_layers * self.hidden_dim
        self.lstm_output_dim = 2 * self.hidden_dim

    def forward(self, batch: Dict):
        # first pass it through the rnn
        rnn_output, (hidden_state, cell_state) = self.rnn(self.dropout(self.bert_model(**batch).last_hidden_state))
        # the shape according to LSTM documentation are:
        # rnn_output: (batch, L, 2 * self.hidden_size)
        # hidden_state, cell_state (2 * num_layers, batch, self.hidden_size)
        return rnn_output, hidden_state, cell_state


class DecoderRNN(nn.Module):
    def __init__(self,
                 token_classifier: nn.Module,
                 emb_dim: int,
                 output_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.2, 
                 go_label: int = -1):
        
        super(DecoderRNN, self).__init__()
        # this label is used as the very first input for the decoder: to start decoding the encoder's hidden state.
        self.go_label = go_label
        self.emb_dim = emb_dim
        # the decoder is a sequence model as well
        self.rnn = nn.LSTM(input_size=emb_dim,
                           hidden_size=output_size,
                           batch_first=True,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=True,
                           )

        # this classifier will convert the output of each LSTM cell to a vocabulary index
        self.classifier = token_classifier

    def forward_step(self, decoder_input, decoder_hs, decoder_cs):
        # this function expects a decoder_input: of shape: (batch_size, 1, 1)
        # decoder_hs should be of the shape (2 * num_layers)
        output, (hidden_states, cell_states) = self.rnn(decoder_input, (decoder_hs, decoder_cs))
        # output at this stage will be (batch_size, 1, self.hidden_size)
        output = self.classifier(output.squeeze(dim=1))
        # output at this point is (batch_Size, vocab_size)
        # hidden_states: (
        # cell_states
        return output, hidden_states, cell_states

    def forward(self,
                encoder_hidden_state,
                encoder_cell_state,
                max_seq_length: int,
                batch_size: int = None,
                target: torch.Tensor = Optional[torch.Tensor],
                device: str = None):

        if target is None and batch_size is None:
            raise ValueError(
                f"either the 'batch_size' or the 'target' arguments must be explicitly passed. Both of them are {None}")

        batch_size = target.dim(0) if target is not None else batch_size
        # the first input is of the size (batch_size, L = 1, input_size = hidden_size)
        # according to the documentation of the nn.embedding layer, padding_idx are initialized to zero_values
        # we are using -1 as the label that represents
        decoder_input = torch.empty(size=(batch_size, 1, 1), dtype=torch.float).fill_(value=self.go_label).to(device)

        decoder_hidden_state = encoder_hidden_state
        decoder_cell_state = encoder_cell_state
        decoder_outputs = []

        for i in range(max_seq_length):
            decoder_output, decoder_hidden_state, decoder_cell_state = self.forward_step(decoder_input,
                                                                                         decoder_hidden_state,
                                                                                         decoder_cell_state)
            # decoder_output will be of the shape (batch_size, num_classes)
            decoder_outputs.append(decoder_output.unsqueeze(dim=1))

            if target is not None:
                # using the target tensor is a technique known as Teacher Forcing
                # the target is expected to be of shape: (batch, L) (as each label is uni dimensional)
                decoder_input = target[:, i].unsqueeze(dim=-1)
            else:
                _, best_prediction = decoder_output.topk(1)
                # detach (so that the error from the previous output is not propagated further to the rest of the sequence)
                # + set to float, as most optimizers work with float data (mainly as input)
                decoder_input = best_prediction.unsqueeze(dim=-1).detach().to(torch.float)

                # the final output should be (batch_size, max_seq_length, classes)
        # each element inside the list is of shape: (batch_size, 1, classes)
        # they should be stacked according to dim = 1
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # reduce to classes predictions
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden_state, decoder_cell_state
