from torch import Tensor


def flatten_final_hidden_state(final_hidden_state: Tensor, batch_first: bool = False):
    """
    Flatten the final hidden state usually obtained from RNN

    final_hidden_state must be a 3 dimension tensor [layers, batch_size, output_dim]
    """
    if batch_first:
        batch_size, layers, _ = final_hidden_state.shape
    else:
        layers, batch_size, _ = final_hidden_state.shape

    if layers == 1:
        return final_hidden_state.squeeze()

    if layers > 1:
        if batch_first:
            return final_hidden_state.reshape(batch_size, -1).contiguous()
        return final_hidden_state.permute(1, 0, 2).reshape(batch_size, -1).contiguous()
