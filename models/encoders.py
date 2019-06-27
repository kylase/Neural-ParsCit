from typing import List, Tuple

import torch
import torch.nn as nn

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMEncoder(nn.Module):
    r"""LSTM-based encoder with embedding layer

    This module can be used for e.g. character-level encoder

    Args:
        vocab_size: int
        embedding_dim: int
        hidden_dim: int
        lstm_layers: int
        lstm_dropout: float
        lstm_bidirectional: bool
        pretrained: Tensor
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 lstm_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 lstm_batch_first: bool = False,
                 padding_index: int = 1,
                 **kwargs):

        super().__init__()

        self.lstm_factor = lstm_layers * (2 ** int(lstm_bidirectional))

        self.hidden_dim = hidden_dim // self.lstm_factor

        self._device = kwargs.get('device', 'cpu')

        self.padding_index = padding_index

        pretrained = kwargs.get('pretrained', None)

        if torch.is_tensor(pretrained):
            self.embeddings = nn.Embedding.from_pretrained(pretrained)
            self.vocab_size, self.embedding_dim = pretrained.shape
        else:
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim,
                           num_layers=lstm_layers,
                           bidirectional=lstm_bidirectional,
                           batch_first=lstm_batch_first)

    def init_hidden_states(self, batch_size: int) -> Tuple[torch.Tensor]:
        return (torch.zeros(self.lstm_factor, batch_size, self.hidden_dim).to(self._device),
                torch.zeros(self.lstm_factor, batch_size, self.hidden_dim).to(self._device))

    def _get_lstm_features(self, embeddings: PackedSequence) -> PackedSequence:
        """
        Args:
            embeddings: PackedSequence
        Return:
            lstm_out: PackedSequence
        """

        lstm_out, hiddens_out = self.rnn(embeddings)

        return lstm_out, hiddens_out

    def forward(self, *inputs: List[torch.Tensor]):
        """
        Args:
            char_inputs: Tensor (batch_size, max_seq_len)

            `batch_size` is the number of words in the sequence
            `max_seq_len` is the max number of characters of the words in the sequence

        Returns:
            features: Tensor
        """
        char_inputs, = inputs

        embed_out = self.embeddings(char_inputs)

        mask = char_inputs.ne(self.padding_index)

        seq_len = torch.sum(mask, dim=1)

        packed_sequence = pack_padded_sequence(embed_out, seq_len,
                                               enforce_sorted=False,
                                               batch_first=True)

        features, hiddens = self._get_lstm_features(packed_sequence)

        return features, hiddens
