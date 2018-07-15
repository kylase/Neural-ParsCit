import pytest

from core.cli import init_parser

def test_file_without_io():
    with pytest.raises(IOError):
        parser = init_parser(['--run', 'file'])

def test_file_without_model():
    with pytest.raises(IOError):
        parser = init_parser(['-i', 'README.md', '-o', 'output.txt', '--run', 'file'])

def test_file_missing_model_files():
    with pytest.raises(Exception):
        parser = init_parser(['--model_path', 'models', '-i', 'README.md', '-o', 'output.txt', '--run', 'file'])
