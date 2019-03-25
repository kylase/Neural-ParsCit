import torch
import torch.nn as nn

from typing import List, Dict

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
        cap_embedding_dim: int
        dropout_rate: float = 0.5
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
                 char_hidden_dim: int, cap_embedding_dim: int, dropout_rate: float = 0.5,
                 word_lstm_layers: int = 1, word_lstm_bidirectional: bool = False,
                 char_lstm_layers: int = 1, char_lstm_bidirectional: bool = False,
                 **kwargs):
        super().__init__()

        self._device = kwargs.get('device', 'cpu')

        char_encoder_params = {
            'vocab_size': char_vocab_size,
            'embedding_dim': char_embedding_dim,
            'hidden_dim': char_hidden_dim,
            'lstm_layers': char_lstm_layers,
            'lstm_bidirectional': char_lstm_bidirectional,
            'lstm_batch_first': True,
            'device': self._device
        }

        # Initialise character-level encoder
        self.char_encoder = LSTMEncoder(**char_encoder_params)

        # Initialise word-level encoder
        self.lstm_factor = word_lstm_layers * (2 ** int(word_lstm_bidirectional))

        self.hidden_dim = word_hidden_dim // self.lstm_factor

        pretrained = kwargs.get('word_pretrained', None)

        if torch.is_tensor(pretrained):
            self.vocab_size = pretrained.shape[0]
            self.embedding_dim = pretrained.shape[1]
            self.embeddings = nn.Embedding.from_pretrained(pretrained)
        else:
            self.vocab_size = word_vocab_size
            self.embedding_dim = word_embedding_dim
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        encoder_input_dim = 1 + self.embedding_dim + \
            self.char_encoder.hidden_dim * self.char_encoder.lstm_factor

        self.cap_embedding = nn.Embedding(cap_embedding_dim, 1)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.encoder = nn.LSTM(encoder_input_dim, self.hidden_dim,
                               num_layers=word_lstm_layers,
                               bidirectional=word_lstm_bidirectional)

        self.hidden_states = None

        self.tag_vocab_size = tag_vocab_size

        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim * self.lstm_factor, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.tag_vocab_size)
        )

        self.decoder = ConditionalRandomField(self.tag_vocab_size)

    @staticmethod
    def generate_mask(tensor: torch.Tensor, mask_value: int = 0) -> torch.Tensor:
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
        if mask_value is None:
            return torch.ones(tensor.size()).long()

        return (tensor != mask_value).long()

    def init_hidden_states(self, batch_size: int) -> None:
        zeros = torch.zeros(self.lstm_factor, batch_size, self.hidden_dim).to(self._device)

        self.hidden_states = (zeros, zeros)

    def _encode_characters(self, char_inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode the characters of a word through the `char_encoder` to obtain
        the final hidden state

        Args:
            char_inputs: Tensor (batch_size, max_seq_len, max_word_len)

        Return:
            final_hidden_state: Tensor ()
        """
        max_seq_len, _ = char_inputs.size()

        # Obtain the final hidden state from the character encoder
        _ = self.char_encoder(char_inputs)

        final_hidden_state, _ = self.char_encoder.hidden_states

        return final_hidden_state.permute(1, 0, 2).contiguous().view(max_seq_len, -1)

    def encode(self, word_inputs: torch.Tensor, char_inputs: torch.Tensor,
               cap_inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode the inputs using the encoder(s)

        Args:
            word_inputs: Tensor (max_seq_len, batch_size)
            char_inputs: Tensor (batch_size, max_seq_len, max_word_len)
            cap_inputs: Tensor (max_seq_len, batch_size)

        Return:
            encoded: Tensor (max_seq_len, batch_size, hidden_dim)
        """
        word_embeddings_out = self.embeddings(word_inputs)
        cap_embedding_out = self.cap_embedding(cap_inputs.t())

        batch_size, max_seq_len, _ = char_inputs.size()

        batch_char_features = []

        for batch_id in range(batch_size):
            self.char_encoder.init_hidden_states(max_seq_len)

            char_features = self._encode_characters(char_inputs[batch_id])

            batch_char_features.append(char_features)

        # Stack on the first dimension (batch_size) to re-create the batch
        # tensor
        batch_char_features = torch.stack(batch_char_features, 1)

        # Concatenated the word embeddings and the character-level embeddings
        # from the LSTM encoder on the second dimension (max_seq_len)
        combined_input = torch.cat((word_embeddings_out, batch_char_features, cap_embedding_out), 2)

        # Apply dropout to combined embeddings
        encoder_input = self.dropout(combined_input)

        encoded, self.hidden_states = self.encoder(encoder_input, self.hidden_states)

        return encoded

    def decode(self, word_inputs: torch.Tensor, char_inputs: torch.Tensor,
               cap_inputs: torch.Tensor, dtype: torch.dtype = torch.long) -> torch.Tensor:
        """
        Decode the encoded sequence

        Args:
            word_inputs: Tensor (max_seq_len, batch_size)
            char_inputs: Tensor (batch_size, max_seq_len, max_word_len)
            cap_inputs: Tensor (max_seq_len, batch_size)
            dtype: torch.dtype
        Return:
            decoded_sequence: Tensor (max_seq_len, batch_size)
        """

        _, batch_size = word_inputs.shape

        self.init_hidden_states(batch_size)

        encoded = self.encode(word_inputs, char_inputs, cap_inputs)

        features = self.feed_forward(encoded)

        mask = self.generate_mask(word_inputs, mask_value=None)

        best_paths = self.decoder.viterbi_tags(features, mask)

        return torch.tensor([x for x, y in best_paths], dtype=dtype)

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        """
        Args:
            word_inputs: Tensor (max_seq_len, batch_size)
            char_inputs: Tensor (batch_size, max_seq_len, max_word_len)
            tag_inputs: Tensor (max_seq_len, batch_size)

        Return:
            outputs: Dict[str, Tensor]
        """
        word_inputs, char_inputs, tag_inputs, cap_inputs = inputs

        # Encode the inputs with character and word-level embeddings
        encoded = self.encode(word_inputs, char_inputs, cap_inputs)

        # Use feed forward layers to reduce the dimension for decoder
        features = self.feed_forward(encoded)

        # Compute the log-likelihood
        mask = self.generate_mask(tag_inputs, mask_value=None)

        log_likelihood = self.decoder(features, tag_inputs, mask)

        best_paths = self.decoder.viterbi_tags(features, mask)

        outputs = {
            'predicted_tags': torch.tensor([x for x, y in best_paths]),
            'mask': mask,
            'loss': -log_likelihood
        }

        return outputs
