import pytest
import torch
from models.encoders import LSTMEncoder

class TestLSTMEncoder:
    VOCAB_SIZE = 20
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
        # Randomly instantiate a 2 strings of sequence length 10
        BATCH_SIZE = 2
        MAX_SEQ_LEN = 10

        input_tensor = torch.randint(low=0, high=self.VOCAB_SIZE - 1,
                                     size=(MAX_SEQ_LEN, BATCH_SIZE))

        for _, model in self.models.items():
            with torch.no_grad():
                encoded = model(input_tensor)

            assert encoded.size() == (MAX_SEQ_LEN, BATCH_SIZE, model.hidden_dim * 2 ** (model.lstm.bidirectional))

    def test_init_hidden_states(self):
        layer = self.models['simple']

        batch_size = 20

        layer.init_hidden_states(batch_size)

        assert layer.hidden_states[0].size() == (1, batch_size, layer.hidden_dim)
        assert layer.hidden_states[1].size() == (1, batch_size, layer.hidden_dim)
