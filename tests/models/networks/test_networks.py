import pytest
import torch
from models.networks import WordCharLSTMCRF


class TestWordCharLSTMCRF:
    WORD_VOCAB_SIZE = 100
    WORD_EMBEDDING_DIM = 50
    WORD_HIDDEN_DIM = 30
    CHAR_VOCAB_SIZE = 20
    CHAR_EMBEDDING_DIM = 50
    CHAR_HIDDEN_DIM = 25
    TAG_VOCAB_SIZE = 10

    models = {}

    def setup(self):
        simple = WordCharLSTMCRF(self.TAG_VOCAB_SIZE, self.WORD_VOCAB_SIZE,
                                 self.WORD_EMBEDDING_DIM, self.WORD_HIDDEN_DIM,
                                 self.CHAR_VOCAB_SIZE, self.CHAR_EMBEDDING_DIM,
                                 self.CHAR_HIDDEN_DIM)

        self.models['simple'] = simple


    def test_init(self):
        assert self.models['simple'].tag_vocab_size == self.TAG_VOCAB_SIZE
        assert self.models['simple'].hidden_dim == self.WORD_HIDDEN_DIM
        assert self.models['simple'].vocab_size == self.WORD_VOCAB_SIZE
        assert self.models['simple'].embedding_dim == self.WORD_EMBEDDING_DIM
        assert self.models['simple'].char_encoder.hidden_dim == self.CHAR_HIDDEN_DIM
        assert self.models['simple'].char_encoder.vocab_size == self.CHAR_VOCAB_SIZE
        assert self.models['simple'].char_encoder.embedding_dim == self.CHAR_EMBEDDING_DIM


    def test_forward(self):
        BATCH_SIZE = 2
        MAX_SEQ_LEN = 10
        MAX_WORD_LENGTH = 6

        word_inputs = torch.randint(low=0, high=self.WORD_VOCAB_SIZE - 1,
                                    size=(MAX_SEQ_LEN, BATCH_SIZE))
        char_inputs = torch.randint(low=0, high=self.CHAR_VOCAB_SIZE - 1,
                                    size=(BATCH_SIZE, MAX_SEQ_LEN, MAX_WORD_LENGTH))
        tag_inputs = torch.randint(low=0, high=self.TAG_VOCAB_SIZE - 1,
                                   size=(MAX_SEQ_LEN, BATCH_SIZE))

        with torch.no_grad():
            for _, model in self.models.items():
                neg_log_likelihood = model(word_inputs, char_inputs, tag_inputs)

                assert neg_log_likelihood.dtype == torch.float


    def test_decode(self):
        BATCH_SIZE = 2
        MAX_SEQ_LEN = 17
        MAX_WORD_LENGTH = 9

        word_inputs = torch.randint(low=0, high=self.WORD_VOCAB_SIZE - 1,
                                    size=(MAX_SEQ_LEN, BATCH_SIZE))
        char_inputs = torch.randint(low=0, high=self.CHAR_VOCAB_SIZE - 1,
                                    size=(BATCH_SIZE, MAX_SEQ_LEN, MAX_WORD_LENGTH))
        with torch.no_grad():
            for _, model in self.models.items():
                decoded_sequence = model.decode(word_inputs, char_inputs)

                assert decoded_sequence.shape == word_inputs.shape


    def test_mask(self):
        unmasked = torch.tensor([[3, 3, 2, 2, 1, 1], [3, 3, 2, 2, 2, 5]])
        expected = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])

        masked = WordCharLSTMCRF.generate_mask(unmasked)

        assert masked.dtype == torch.long
        assert torch.allclose(masked, expected)


    def test_init_hidden_states(self):
        model = self.models['simple']

        batch_size = 20

        model.init_hidden_states(batch_size)

        assert model.hidden_states[0].size() == (1, batch_size, model.hidden_dim)
        assert model.hidden_states[1].size() == (1, batch_size, model.hidden_dim)

        max_word_length = 27

        model.char_encoder.init_hidden_states(max_word_length)

        assert model.char_encoder.hidden_states[0].size() == (1, max_word_length, model.char_encoder.hidden_dim)
        assert model.char_encoder.hidden_states[1].size() == (1, max_word_length, model.char_encoder.hidden_dim)
