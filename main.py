import os
import logging

from random import randint
from collections import namedtuple

import torch
import torch.optim as optim
from torch.nn import NLLLoss

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from tqdm import tqdm

from core.dataset import CoraWithFeatures
from torchtext.data import Field, LabelField, NestedField, BucketIterator
from torchtext.vocab import Vectors

from models.networks import WordCharLSTMCRF

from cli.parser import create_parser


class ACM(Vectors):
    url_base = 'http://wing.comp.nus.edu.sg/acm.500d.txt'

    def __init__(self, **kwargs):
        url = self.url_base
        name = os.path.basename(url)
        super().__init__(name, url=url, **kwargs)


def flatten_examples(examples):
    y_true = []
    y_pred = []

    for example in examples:
        y_true.extend(example.truth)
        y_pred.extend(example.predicted)

    return y_true, y_pred


def report(dataset, model: torch.nn.Module, vocab, device):
    logger = logging.getLogger(__name__)
    logger.info('Evaluating...')

    example = namedtuple('Example', ['predicted', 'truth'])
    examples = []

    with torch.no_grad():
        for batch in iter(dataset):
            seq_len, batch_size = batch.text.shape

            model.init_hidden_states(batch_size)

            text = batch.text.to(device)
            char = batch.char.to(device)

            decoded_sequence = model.decode(text, char)

            examples.append(example(decoded_sequence.numpy(), batch.label.numpy()))

    # Print a random example
    rand_idx = randint(0, len(examples))

    # TODO Pad the prefix to align the sequences
    logger.info('Getting a random example %d.', rand_idx)
    # print(f"Actual string: {dataset[rand_idx].text}")
    print(f"Ground Truth Sequence: {examples[rand_idx].truth.transpose()}")
    print(f"Predicted Sequence: {examples[rand_idx].predicted.transpose()}")

    true, pred = flatten_examples(examples)
    # Print overall performance
    print(classification_report(true, pred, labels=vocab.vocab.itos))
    print(f"Micro F1: {f1_score(true, pred, average='micro')}; Macro F1: {f1_score(true, pred, average='macro')}")


def main():
    parser = create_parser()

    parser.parse_args()

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info("Using %s for training.", device)

    WORD = Field()
    LABEL = Field()
    CHAR_NESTING = Field(tokenize=list)
    CHAR = NestedField(CHAR_NESTING)

    fields = [(('text', 'char'), (WORD, CHAR))] + [(None, None)] * 22 + [('label', LABEL)]

    train, val, test = CoraWithFeatures.splits(fields=fields)

    logger.info('Building vocabularies...')

    WORD.build_vocab(train.text, val.text, test.text, vectors=[ACM()])
    LABEL.build_vocab(train.label, val.label, test.label)
    CHAR.build_vocab(train.char, val.char, test.char)

    params = {
        'tag_vocab_size': len(LABEL.vocab),
        'word_vocab_size': len(WORD.vocab),
        'word_embedding_dim': 500,
        'word_hidden_dim': 100,
        'char_vocab_size': len(CHAR.vocab),
        'char_embedding_dim': 50,
        'char_hidden_dim': 50,
        'pretrained': WORD.vocab.vectors,
        'word_lstm_bidirectional': True,
        'char_lstm_bidirectional': True,
        'device': device
    }

    logger.info('Instantiating model with parameters...')

    model = WordCharLSTMCRF(**params)

    model.to(device)

    train_iter, val_iter, test_iter = BucketIterator.splits((train, val, test), batch_sizes=(1, 1, 1))

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss_tracker = []

    logger.info('Training model...')

    for epoch in tqdm(range(1), desc='Epoch'):
        for batch in tqdm(train_iter, total=len(train_iter), desc="Batch"):
            model.zero_grad()

            seq_len, batch_size = batch.text.shape

            model.init_hidden_states(batch_size)

            text = batch.text.to(device)
            char = batch.char.to(device)
            label = batch.label.to(device)

            score = model(text, char, label)

            loss = score

            loss.backward()
            optimizer.step()

            loss_tracker.append((epoch, loss.item()))

    logger.info('Training completed.')

    report(test_iter, model, LABEL, device)


if __name__ == '__main__':
    main()
