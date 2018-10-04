import pytest
from torchtext.data import Field, NestedField
from parscit.dataset import Cora

class TestCora(object):
    WORD = Field(init_token='<bos>', eos_token='<eos>')
    LABEL = Field(init_token='<bos>', eos_token='<eos>')
    fields = [('text', WORD)] + [(None, None)] * 22 + [('label', LABEL)]

    def test_splits(self):
        train, val, test = Cora.splits(fields=self.fields)
        total_len = len(train) + len(val) + len(test)

        assert len(train) == 0.8 * total_len
        assert len(val) == 0.1 * total_len
        assert len(test) == 0.1 * total_len
