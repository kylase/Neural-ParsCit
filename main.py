import json
import logging

from random import randint
from collections import namedtuple, defaultdict, deque

import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_value_

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorboardX import SummaryWriter

from tqdm import tqdm

from core.dataset import CoraWithFeatures, ETD
from torchtext.data import Field, NestedField, BucketIterator
from dataset.fields import cap_feature
from dataset.vocab import AugmentedACM

from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import GloVe

from models.networks import WordCharLSTMCRF

# from cli.parser import create_parser

def flatten_examples(examples):
    y_true = []
    y_pred = []

    for example in examples:
        y_true.extend(example.truth.flatten())
        y_pred.extend(example.predicted.flatten())

    return y_true, y_pred

def report(dataset, params, state, vocab, verbose=True, print_random_sequence=True):
    example = namedtuple('Example', ['text', 'predicted', 'truth'])
    examples = []

    model = WordCharLSTMCRF(**params)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        for batch in iter(dataset):
        # for batch in tqdm(dataset, total=len(dataset), desc="Evaluation"):
            seq_len, batch_size = batch.text.shape

            model.init_hidden_states(batch_size)

            text = batch.text
            char = batch.char
            cap = batch.cap

            decoded_sequence = model.decode(text, char, cap)

            examples.append(example(text.numpy(), decoded_sequence.numpy(), batch.label.numpy()))

    true, pred = flatten_examples(examples)

    if print_random_sequence:
        # TODO Pad the prefix to align the sequences
        rand_idx = randint(0, len(examples) - 1)
        # Print a random example
        print(f"Ground Truth Sequence: {examples[rand_idx].truth.transpose()}")
        print(f"   Predicted Sequence: {examples[rand_idx].predicted.transpose()}")

        # Print overall performance
        # print(f"Micro F1: {f1_score(true, pred, average='micro')}; Macro F1: {f1_score(true, pred, average='macro')}")
        # print(classification_report(true, pred, labels=range(len(vocab.vocab)),
        #                             target_names=vocab.vocab.itos, digits=4))

    report = classification_report(true, pred,
                                   labels=range(len(vocab.vocab)),
                                   target_names=vocab.vocab.itos,
                                   digits=4,
                                   output_dict=not verbose)

    if verbose:
        print(report)
        print(confusion_matrix(true, pred))

        return examples
    else:
        reporting_metrics = defaultdict(dict)

        for label, metrics in report.items():
            for metric, value in metrics.items():
                reporting_metrics[metric][label] = value

        return reporting_metrics



def main():
    # parser = create_parser()

    # parser.parse_args()

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARN)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info("Using %s for training.", device)

    # torch.manual_seed(0)
    import re
    is_digit = re.compile("\d")
    def zeroify(chars):
        return list(map(lambda char: '0' if is_digit.search(char) else char, chars))

    WORD = Field()
    LABEL = Field(unk_token=None, pad_token=None, is_target=True)
    CHAR_NESTING = Field(tokenize=list, preprocessing=zeroify, pad_token="<p>")
    CHAR = NestedField(CHAR_NESTING)
    CAP_NESTING = Field(preprocessing=cap_feature, sequential=False, use_vocab=False)
    CAP = NestedField(CAP_NESTING)

    # CORA_FIELDS = [(('text', 'char', 'cap'), (WORD, CHAR, CAP))] + [(None, None)] * 22 + [('label', LABEL)]

    # train, test = CoraWithFeatures.splits(fields=CORA_FIELDS, k=0, test_frac=0.1)

    # CoNLL 2003
    fields = [(('text', 'char', 'cap'), (WORD, CHAR, CAP)), (None, None), (None, None), ('label', LABEL)]

    train, val, test = SequenceTaggingDataset.splits(path='.data/conll2003',
                                                     train='eng.train',
                                                     validation='eng.testa',
                                                     test='eng.testb',
                                                     separator=' ',
                                                     fields=fields)

    # etd_fields = [(('text', 'char'), (WORD, CHAR))] + [('label', LABEL)]
    #
    # train, val, test = ETD.splits(etd_fields)
    logger.info('Building vocabularies...')

    WORD.build_vocab(train.text, train.text, test.text, vectors=[GloVe(name='6B', dim='50')])
    LABEL.build_vocab(train.label, train.label, test.label)
    CHAR.build_vocab(train.char, train.char, test.char)

    # WORD.build_vocab(train.text, train.text, test.text, vectors=[AugmentedACM()])
    # LABEL.build_vocab(train.label, train.label, test.label)
    # CHAR.build_vocab(train.char, train.char, test.char)

    params = {
        'tag_vocab_size': len(LABEL.vocab),
        'word_vocab_size': len(WORD.vocab),
        'word_embedding_dim': 500,
        'word_hidden_dim': 200,
        'char_vocab_size': len(CHAR.vocab),
        'char_embedding_dim': 25,
        'char_hidden_dim': 50,
        'cap_embedding_dim': 4,
        'dropout_rate': 0.5,
        'pretrained': WORD.vocab.vectors,
        'word_lstm_bidirectional': True,
        'char_lstm_bidirectional': False,
        'device': device
    }

    logger.info('Instantiating model with parameters...')

    # for k, v in model.named_parameters():
    #     print(k, v)

    def params_to_str(params):
        params_keys = ['tag_vocab_size',
                       'word_vocab_size',
                       'word_embedding_dim',
                       'word_hidden_dim',
                       'char_vocab_size',
                       'char_embedding_dim',
                       'char_hidden_dim',
                       'word_lstm_bidirectional',
                       'char_lstm_bidirectional']

        pretrained = params.get('pretrained')

        return json.dumps({k: params[k] for k in params_keys}, separators=(',', ':'), sort_keys=True)

    model = WordCharLSTMCRF(**params)
    model.to(device)

    # Initialise weights
    for module in model.modules():
        if isinstance(module, (nn.Linear)):
            nn.init.uniform_(module.weight, a=-1, b=1)
            # nn.init.xavier_uniform_(m.weight)

    train_iter, test_iter = BucketIterator.splits((train, test),
                                                  batch_sizes=(1, 1, 1),
                                                  shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.005)
    # writer = SummaryWriter(log_dir=f"logs/expt:{params_to_str(params)}")
    writer = SummaryWriter(log_dir=f"logs/expt")
    # logger.info("Training model using %i fold...", fold_idx)

    log_frequency = 50
    counter = 0

    best_model_state = None
    highest_loss = 1e6

    for _ in tqdm(range(1), desc=f"Epoch"):
        train_length = len(train_iter)
        losses = deque(maxlen=min(log_frequency, round(train_length/log_frequency)))

        for batch in tqdm(train_iter, total=train_length, desc="Batch"):
            counter += 1
            model.zero_grad()

            _, batch_size = batch.text.shape

            model.init_hidden_states(batch_size)

            text = batch.text.to(device)
            char = batch.char.to(device)
            label = batch.label.to(device)
            cap = batch.cap.to(device)

            outputs = model(text, char, label, cap)

            loss = outputs['loss']

            losses.append(outputs['loss'].item())

            average_loss = sum(losses)/len(losses)

            if average_loss < highest_loss:
                logger.info("New lowest loss: %f", average_loss)
                highest_loss = average_loss
                logger.info("Saving model state...")
                best_model_state = model.state_dict()

            if not counter % max(log_frequency, round(train_length/log_frequency)) or counter == train_length:
                # reports = {
                #     'validation': report(val_iter, params, model.state_dict(), LABEL, verbose=False, print_random_sequence=False),
                #     'test': report(test_iter, params, model.state_dict(), LABEL, verbose=False, print_random_sequence=False)
                # }

                # writer.add_scalars('metrics', reports, counter)
                # for s, r in reports.items():
                #     for metric, labels_perf in r.items():
                #         writer.add_scalars(f"{s}/{metric}", labels_perf, counter)
                writer.add_scalar('training/metrics/average_negative_log_likelihood', sum(losses)/len(losses), counter)

            loss.backward()
            clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()

    logger.info('Training completed.')
    # torch.save(best_model_state, 'best_model.pt')

    print(f"Training:")
    report(train_iter, params.copy(), best_model_state, LABEL)
    # print(f"Validation:")
    # report(val_iter, params.copy(), best_model_state, LABEL)
    print(f"Test:")
    examples = report(test_iter, params.copy(), best_model_state, LABEL)

    # writer.export_scalars_to_json('test.json')
    writer.close()
if __name__ == '__main__':
    main()
