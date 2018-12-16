import pytest
import torch
from models.encoders import LSTMEncoder

class TestLSTMEncoder:
    def test_init(self):
        simple_encoder = LSTMEncoder(vocab_size=20,
                                     embedding_dim=50,
                                     hidden_dim=10)

        bidirectional_encoder = LSTMEncoder(vocab_size=20,
                                            embedding_dim=50,
                                            hidden_dim=10,
                                            lstm_bidirectional=True)

        two_layers_bidirectional_encoder = LSTMEncoder(vocab_size=20,
                                                       embedding_dim=50,
                                                       hidden_dim=16,
                                                       lstm_layers=2,
                                                       lstm_bidirectional=True)

        assert simple_encoder.hidden_dim == 10
        assert bidirectional_encoder.hidden_dim == 5
        assert two_layers_bidirectional_encoder.hidden_dim == 4

    def test_init_with_pretained(self):
        fake_embeddings = torch.rand(15, 20)

        params = {
            'vocab_size': 20,
            'embedding_dim': 50,
            'hidden_dim': 10
        }

        encoder = LSTMEncoder(pretrained=fake_embeddings, **params)

        assert encoder.vocab_size == 15
        assert encoder.embedding_dim == 20
