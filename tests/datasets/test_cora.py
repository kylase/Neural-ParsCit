import pytest
from torchtext.data import Field, NestedField
from core.dataset import CoraWithFeatures

class TestCoraDataset:
    def setup(self):
        WORD = Field(init_token='<bos>', eos_token='<eos>')
        CHAR_NESTING = Field(tokenize=list, init_token='<bos>', eos_token='<eos>')
        CHAR = NestedField(CHAR_NESTING, init_token='<bos>', eos_token='<eos>')
        ENTITY = Field(init_token='<bos>', eos_token='<eos>')

        self.fields = [(('text', 'char'), (WORD, CHAR))] + [(None, None)] * 22 + [('entity', ENTITY)]

    def test_splits(self):
        train, val, test = CoraWithFeatures.splits(self.fields)

        assert len(train) == 400
        assert len(val) == 50
        assert len(test) == 50
