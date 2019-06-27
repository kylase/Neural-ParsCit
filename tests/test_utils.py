import pytest
import torch

from utils import flatten_final_hidden_state

class TestFlattenEncoderOutput:
    def setup(self):
        # Tensor dimension = [num_layers, batch_size, hidden_dim]
        self.single_layer_data = torch.rand(1, 10, 5)
        self.n_layer_data = torch.rand(2, 10, 12)

        # Tensor dimension = [batch_size, num_layers, hidden_dim]
        self.batch_first_single_layer_data = torch.rand(10, 1, 8)
        self.batch_first_n_layer_data = torch.rand(10, 2, 9)

    def test_flatten_single_layer(self):
        flattened_data = flatten_final_hidden_state(self.single_layer_data)
        expected_shape = (self.single_layer_data.shape[1], self.single_layer_data.shape[2])

        assert flattened_data.shape == expected_shape

        assert flattened_data.is_contiguous() is True

    def test_flatten_n_layers(self):
        flattened_data = flatten_final_hidden_state(self.n_layer_data)

        expected_shape = (self.n_layer_data.shape[1],
                          self.n_layer_data.shape[2] * self.n_layer_data.shape[0])

        assert flattened_data.shape == expected_shape

        for batch_id in range(self.n_layer_data.shape[1]):
            expected_tensor = torch.cat([self.n_layer_data[0, batch_id], self.n_layer_data[1, batch_id]])
            assert flattened_data[batch_id].equal(expected_tensor) is True

        assert flattened_data.is_contiguous() is True

    def test_flatten_batch_first_single_layer(self):
        flattened_data = flatten_final_hidden_state(self.batch_first_single_layer_data, batch_first=True)
        expected_shape = (self.batch_first_single_layer_data.shape[0], self.batch_first_single_layer_data.shape[2])

        assert flattened_data.shape == expected_shape

        assert flattened_data.is_contiguous() is True

    def test_flatten_batch_first_n_layer(self):
        flattened_data = flatten_final_hidden_state(self.batch_first_n_layer_data, batch_first=True)

        expected_shape = (self.batch_first_n_layer_data.shape[0],
                          self.batch_first_n_layer_data.shape[2] * self.batch_first_n_layer_data.shape[1])

        assert flattened_data.shape == expected_shape

        for batch_id in range(self.batch_first_n_layer_data.shape[0]):
            expected_tensor = torch.cat([self.batch_first_n_layer_data[batch_id, 0],
                                         self.batch_first_n_layer_data[batch_id, 1]])
            assert flattened_data[batch_id].equal(expected_tensor) is True

        assert flattened_data.is_contiguous() is True
