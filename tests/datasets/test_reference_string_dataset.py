import pytest

from pathlib import Path

from dataset import ReferenceStringDataset


class TestReferenceStringDataset:
    def setUp(self):
        fixture_path = Path('tests/fixtures/reference_strings')
        self.samples = map(lambda f: f.open(encoding='utf-8').readlines(), fixture_path.glob('sample.*.txt'))

        self.dataset = ReferenceStringDataset(fixture_path, 'sample.*.txt', style='sample')

    def test_len(self):
        assert len(self.dataset) == 25

    def test_getitem(self):
        assert self.dataset[0] == self.samples[0][0].strip()
        assert self.dataset[-1] == self.samples[1][-1].strip()
        # assert self.dataset[26]