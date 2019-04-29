import random
from collections import namedtuple

from sklearn.model_selection import KFold
from torchtext.data import Dataset
from torchtext.datasets import SequenceTaggingDataset

CrossValidationDatasets = namedtuple('CrossValidationDatasets', ['train', 'val'])

class CoraWithFeatures(SequenceTaggingDataset):
    """
    This loads the Cora training dataset that is used to train the CRF model.

    Since there is only training data, the training data will be split to
    train, test in the ratio of 9:1 (by default). The training data will be
    further split k-folds.
    """

    urls = ['https://raw.githubusercontent.com/knmnyn/ParsCit/master/crfpp/traindata/cora.train']
    dirname = ''
    name = 'cora'

    @classmethod
    def splits(cls, fields, root='.data', train='cora.train',
               test_frac=0.1, k=10, shuffle=True, **kwargs):
        """
        Splits the dataset to training and test

        if cross validation (`k` > 0), then split the training to training and
        validation using sklearn.model_selection.KFold.

        Return:
            training ()
            test (Dataset)
        """
        train, = super().splits(fields=fields,
                                root=root,
                                train=train,
                                separator=' ',
                                **kwargs)

        # HACK: Saving the sort key function as the split() call removes it
        sort_key = train.sort_key

        # Now split the train set
        # Force a random seed to make the split deterministic
        random.seed(kwargs.get('seed', 0))
        split_ratios = [1 - test_frac, test_frac]
        train, test = train.split(split_ratios, random_state=random.getstate())
        # Reset the seed
        random.seed()

        if k > 0:
            shuffling_kfolds = KFold(n_splits=k, shuffle=shuffle)
            folds = []
            for train_idx, val_idx in shuffling_kfolds.split(train.examples):
                random.shuffle(train_idx)
                random.shuffle(val_idx)

                tr = Dataset(list(map(train.__getitem__, train_idx)), fields)
                val = Dataset(list(map(train.__getitem__, val_idx)), fields)
                # HACK: Set the sort key
                tr.sort_key = sort_key
                val.sort_key = sort_key

                folds.append(CrossValidationDatasets(tr, val))

            return tuple(folds), test

        return train, test


class ETD(SequenceTaggingDataset):
    urls = ['http://wing.comp.nus.edu.sg/etd.tar.gz']
    dirname = ''
    name = 'etd'

    @classmethod
    def splits(cls, fields, root='.data', train='train.txt', validation='test.txt',
               test='val.txt', **kwargs):
        return super().splits(fields=fields,
                              root=root,
                              train=train,
                              validation=validation,
                              test=test,
                              separator='\t',
                              **kwargs)

class CoNLL2003(SequenceTaggingDataset):
    urls = ['https://wing.comp.nus.edu.sg/ner/conll2003/eng.train',
            'https://wing.comp.nus.edu.sg/ner/conll2003/eng.testa',
            'https://wing.comp.nus.edu.sg/ner/conll2003/eng.testb']
    dirname = ''
    name = 'conll2003'

    @classmethod
    def splits(cls, fields, root='.data', train='eng.train', test='eng.testa', **kwargs):
        train, test = super().splits(fields=fields,
                                     root=root,
                                     train=train,
                                     test=test,
                                     separator='\t',
                                     **kwargs)

        return train, test
