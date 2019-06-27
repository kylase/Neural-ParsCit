import pytest
import torch

from typing import Tuple
from torch.nn.utils.rnn import PackedSequence, pad_sequence

from models.encoders import LSTMEncoder


class TestLSTMEncoder:
    VOCAB_SIZE = ord('z') - ord('a') + 1
    EMBEDDING_DIM = 50

    def setup(self):
        simple = LSTMEncoder(vocab_size=self.VOCAB_SIZE,
                             embedding_dim=self.EMBEDDING_DIM,
                             hidden_dim=10)

        bidirectional = LSTMEncoder(vocab_size=self.VOCAB_SIZE,
                                    embedding_dim=self.EMBEDDING_DIM,
                                    hidden_dim=10,
                                    lstm_bidirectional=True)

        two_layers_bidirectional = LSTMEncoder(vocab_size=self.VOCAB_SIZE,
                                               embedding_dim=self.EMBEDDING_DIM,
                                               hidden_dim=16,
                                               lstm_layers=2,
                                               lstm_bidirectional=True)

        params = {
            'vocab_size': self.VOCAB_SIZE,
            'embedding_dim': self.EMBEDDING_DIM,
            'hidden_dim': 12
        }

        fake_embeddings = torch.rand(self.VOCAB_SIZE, params.get('embedding_dim'))

        simple_with_pretrained = LSTMEncoder(pretrained=fake_embeddings, **params)

        self.models = {
            'simple': simple,
            'bidirectional': bidirectional,
            'two_layers_bidirectional': two_layers_bidirectional,
            'simple_with_pretrained': simple_with_pretrained
        }

    def test_init(self):
        assert self.models['simple'].hidden_dim == 10
        assert self.models['bidirectional'].hidden_dim == 5
        assert self.models['two_layers_bidirectional'].hidden_dim == 4
        assert self.models['simple_with_pretrained'].hidden_dim == 12

        assert self.models['simple'].vocab_size == self.VOCAB_SIZE
        assert self.models['bidirectional'].vocab_size == self.VOCAB_SIZE
        assert self.models['two_layers_bidirectional'].vocab_size == self.VOCAB_SIZE
        assert self.models['simple_with_pretrained'].vocab_size == self.VOCAB_SIZE

        assert self.models['simple'].embedding_dim == self.EMBEDDING_DIM
        assert self.models['bidirectional'].embedding_dim == self.EMBEDDING_DIM
        assert self.models['two_layers_bidirectional'].embedding_dim == self.EMBEDDING_DIM
        assert self.models['simple_with_pretrained'].embedding_dim == self.EMBEDDING_DIM

    def test_forward(self):
        # Test data
        # a quick brown fox jumps over the wall
        test_string = "a quick brown fox jumps over the wall"
        char_indices = [list(map(lambda w: ord(w) - ord('a') + 1, word)) \
                        for word in test_string.split(" ")]

        padded_sequence = pad_sequence([torch.tensor(c) for c in char_indices]).contiguous()

        # pad_sequence return tensor of shape [max_seq_len, batch_size]
        test_shape = (max([len(w) for w in test_string.split(" ")]),
                      len(test_string.split(" ")))
        assert padded_sequence.shape == test_shape

        for _, model in self.models.items():
            with torch.no_grad():
                encoded, hidden_states = model(padded_sequence)

            assert isinstance(encoded, PackedSequence)
            assert isinstance(hidden_states, Tuple)

            # Check the shape of the final hidden state
            assert hidden_states[0].shape == (model.lstm_factor, test_shape[0], model.hidden_dim)

    def test_init_hidden_states(self):
        layer = self.models['simple']

        batch_size = 20

        hidden_states = layer.init_hidden_states(batch_size)

        assert hidden_states[0].shape == (1, batch_size, layer.hidden_dim)
        assert hidden_states[1].shape == (1, batch_size, layer.hidden_dim)
