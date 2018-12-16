import random
from torchtext.datasets import SequenceTaggingDataset

class CoraWithFeatures(SequenceTaggingDataset):
    """
    This loads the Cora training dataset that is used to train the CRF model.

    Since there is only training data, the training data will be split to
    train, validation, test in the ratio of 8:1:1 (by default)
    """

    urls = ['https://raw.githubusercontent.com/knmnyn/ParsCit/master/crfpp/traindata/cora.train']
    dirname = ''
    name = 'cora'

    @classmethod
    def splits(cls, fields, root='.data', train='cora.train',
               validation_frac=0.1, test_frac=0.1,
               **kwargs):
        train, = super().splits(fields=fields,
                                root=root,
                                train=train,
                                separator=' ',
                                **kwargs)

        # HACK: Saving the sort key function as the split() call removes it
        sort_key = train.sort_key

        # Now split the train set
        # Force a random seed to make the split deterministic
        random.seed(0)
        split_ratios = [1 - validation_frac - test_frac, validation_frac, test_frac]
        train, val, test = train.split(split_ratios, random_state=random.getstate())
        # Reset the seed
        random.seed()

        # HACK: Set the sort key
        train.sort_key = sort_key
        val.sort_key = sort_key

        return train, val, test
