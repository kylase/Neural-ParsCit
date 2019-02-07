import torch
import torch.nn as nn

from typing import List

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
                 lstm_dropout: float = 0.0,
                 lstm_bidirectional: bool = False,
                 lstm_batch_first: bool = False,
                 **kwargs):

        super().__init__()

        self.lstm_factor = lstm_layers * (2 ** int(lstm_bidirectional))

        self.hidden_dim = hidden_dim // self.lstm_factor

        self._device = kwargs.get('device', 'cpu')

        pretrained = kwargs.get('pretrained', None)

        if torch.is_tensor(pretrained):
            self.embeddings = nn.Embedding.from_pretrained(pretrained)
            self.embedding_dim = pretrained.shape[1]
            self.vocab_size = pretrained.shape[0]
        else:
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers=lstm_layers, dropout=lstm_dropout,
                            bidirectional=lstm_bidirectional,
                            batch_first=lstm_batch_first)

        self.hidden_states = None

    def init_hidden_states(self, batch_size: int):
        zeros = torch.zeros(self.lstm_factor, batch_size, self.hidden_dim)

        self.hidden_states = (zeros, zeros)

    def _get_lstm_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return:
            lstm_out of shape (seq_len, batch, num_directions * hidden_size)
        """

        lstm_out, self.hidden_states = self.lstm(features, self.hidden_states)

        return lstm_out

    def forward(self, *inputs: List[torch.Tensor]):
        """
        Args:
            inputs: List[Tensor]
        """
        embed_out = self.embeddings(inputs[0])

        lstm_features = self._get_lstm_features(embed_out)

        return lstm_features
