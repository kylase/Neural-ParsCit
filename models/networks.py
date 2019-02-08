import torch
import torch.nn as nn

from typing import List

from models.encoders import LSTMEncoder
from crf.modules import ConditionalRandomField

class WordCharLSTMCRF(nn.Module):
    r"""
    Word and character-level LSTM encoders with CRF decoder [Lample, 2016]

    Args:
        tag_vocab_size: int
        word_vocab_size: int
        word_embedding_dim: int
        word_hidden_dim: int
        char_vocab_size: int
        char_embedding_dim: int
        char_hidden_dim: int
        word_lstm_layers: int = 1
        word_lstm_bidirectional: bool = False
        char_lstm_layers: int = 1
        char_lstm_dropout: float = 0.0
        char_lstm_bidirectional: bool = False
        pretrained: Tensor
    """
    def __init__(self, tag_vocab_size: int, word_vocab_size: int,
                 word_embedding_dim: int, word_hidden_dim: int,
                 char_vocab_size: int, char_embedding_dim: int,
                 char_hidden_dim: int, word_lstm_layers: int = 1,
                 word_lstm_bidirectional: bool = False, char_lstm_layers: int = 1,
                 char_lstm_dropout: float = 0.0, char_lstm_bidirectional: bool = False,
                 **kwargs):
        super().__init__()

        # Initialise character-level encoder
        self.char_encoder = LSTMEncoder(char_vocab_size, char_embedding_dim,
                                        char_hidden_dim,
                                        lstm_layers=char_lstm_layers,
                                        lstm_dropout=char_lstm_dropout,
                                        lstm_bidirectional=char_lstm_bidirectional,
                                        lstm_batch_first=True)

        # Initialise word-level encoder
        self.lstm_factor = word_lstm_layers * (2 ** int(word_lstm_bidirectional))

        self.hidden_dim = word_hidden_dim // self.lstm_factor

        self._device = kwargs.get('device', 'cpu')

        pretrained = kwargs.get('word_pretrained', None)

        if torch.is_tensor(pretrained):
            self.vocab_size = pretrained.shape[0]
            self.embedding_dim = pretrained.shape[1]
            self.embeddings = nn.Embedding.from_pretrained(pretrained)
        else:
            self.vocab_size = word_vocab_size
            self.embedding_dim = word_embedding_dim
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        encoder_input_dim = self.embedding_dim + \
            self.char_encoder.hidden_dim * self.char_encoder.lstm_factor

        self.encoder = nn.LSTM(encoder_input_dim, self.hidden_dim,
                               num_layers=word_lstm_layers,
                               bidirectional=word_lstm_bidirectional)

        self.hidden_states = None

        self.tag_vocab_size = tag_vocab_size

        self.hidden_layer = nn.Linear(self.hidden_dim * self.lstm_factor, self.tag_vocab_size)

        self.decoder = ConditionalRandomField(self.tag_vocab_size)

    @staticmethod
    def generate_mask(tensor: torch.Tensor, mask_value: int = 1) -> torch.Tensor:
        """
        Convert the elements which is equal to mask_value (index of padding)
        to 0 and the rest to 1

        Args:
            tensor: torch.Tensor
            mask_value (int) the value which will be 0 after masking (usually
            the index of the padding)

        Return:
            mask: LongTensor
        """
        return (tensor != mask_value).long()

    def init_hidden_states(self, batch_size: int) -> None:
        zeros = torch.zeros(self.lstm_factor, batch_size, self.hidden_dim)

        self.hidden_states = (zeros, zeros)

    def _char_features(self, char_inputs):
        max_seq_len, _ = char_inputs.size()

        # Obtain the final hidden state from the character encoder
        _ = self.char_encoder(char_inputs)

        final_hidden_state, _ = self.char_encoder.hidden_states

        return final_hidden_state.permute(1, 0, 2).contiguous().view(max_seq_len, -1)

    def encode(self, word_inputs, char_inputs):
        """
        Encode the inputs using the encoder(s)

        Return:
            encoded: Tensor (max_seq_len, batch_size, hidden_dim)
        """
        word_embeddings_out = self.embeddings(word_inputs)

        batch_size, max_seq_len, _ = char_inputs.size()

        batch_char_features = []

        for batch_id in range(batch_size):
            self.char_encoder.init_hidden_states(max_seq_len)

            char_features = self._char_features(char_inputs[batch_id])

            batch_char_features.append(char_features)

        # Stack on the first dimension (batch_size) to re-create the batch
        # tensor
        batch_char_features = torch.stack(batch_char_features, 1)

        # Concatenated the word embeddings and the character-level embeddings
        # from the LSTM encoder on the second dimension (max_seq_len)
        combined_input = torch.cat((word_embeddings_out, batch_char_features), 2)

        encoded, self.hidden_states = self.encoder(combined_input, self.hidden_states)

        return encoded

    def decode(self, word_inputs: torch.Tensor, char_inputs: torch.Tensor,
               dtype: torch.dtype = torch.long):
        """
        Decode the encode sequence

        Args:
            word_inputs: Tensor (max_seq_len, batch_size)
            char_inputs: Tensor (batch_size, max_seq_len, max_word_len)
            dtype: torch.dtype
        Return:
            decoded_sequence: Tensor (max_seq_len, batch_size)
        """

        _, batch_size = word_inputs.shape

        self.init_hidden_states(batch_size)

        encoded = self.encode(word_inputs, char_inputs)

        features = self.hidden_layer(encoded)

        mask = self.generate_mask(word_inputs)

        best_paths = self.decoder.viterbi_tags(features, mask)

        return torch.tensor([x for x, y in best_paths], dtype=dtype)

    def forward(self, *inputs) -> float:
        """
        Args:
            word_inputs: Tensor (max_seq_len, batch_size)
            char_inputs: Tensor (batch_size, max_seq_len, max_word_len)
            tag_inputs: Tensor (max_seq_len, batch_size)

        Return:
            negative_log_likelihood: FloatTensor
        """
        word_inputs, char_inputs, tag_inputs = inputs

        # Encode the inputs with character and word-level embeddings
        encoded = self.encode(word_inputs, char_inputs)

        # Use linear layer to reduce the dimension for decoder layer
        decoder_inputs = self.hidden_layer(encoded)

        # Compute the log-likelihood
        mask = self.generate_mask(tag_inputs)
        log_likelihood = self.decoder(decoder_inputs, tag_inputs, mask)

        return -1.0 * log_likelihood
