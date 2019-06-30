import linecache
import logging

from collections import OrderedDict
from pathlib import Path

from torch.utils.data import Dataset


class ReferenceStringDataset(Dataset):
    def __init__(self, directory, filename, encoding='utf-8', **kwargs):
        """

        Args:
            directory: path which the file(s) reside(s)
            filename: filename (glob-able)
            encoding: 'utf-8'
            **kwargs:
        """

        self.style = kwargs.get('style', None)

        self.max_line_index_in_files = OrderedDict()

        line_counter = 0

        for fh in Path(directory).glob(filename):
            logging.info("Reading %s...", fh)
            line_counter += len([line for line in fh.open(encoding=encoding)])
            self.max_line_index_in_files.update(OrderedDict({str(fh): line_counter}))

    def __len__(self):
        return list(self.max_line_index_in_files.values())[-1]

    def __getitem__(self, item):
        if item > len(self):
            raise IndexError()

        if item < 0:
            item = len(self) - item

        offset = 0

        for filename, max_line_index in self.max_line_index_in_files.items():
            if item < max_line_index:
                return linecache.getline(filename, item - offset + 1).strip()

            offset += max_line_index


