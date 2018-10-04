import random
from torchtext.datasets import SequenceTaggingDataset

class Cora(SequenceTaggingDataset):
    """
    Since there is only training data, the training data will be split to
    train, validation, test in the ratio of 8:1:1

    `fields` consists of 25 objects
    """

    urls = ['https://raw.githubusercontent.com/knmnyn/ParsCit/master/crfpp/traindata/cora.train']
    dirname = ''
    name = 'cora'

    @classmethod
    def splits(cls, fields, root=".data", train="cora.train", validation_frac=0.1, test_frac=0.1,
               **kwargs):
        train, = super(Cora, cls).splits(fields=fields, root=root, train=train,
                                         separator=' ', **kwargs)

        sort_key = train.sort_key

        random.seed(0)

        train, test = train.split(1 - (validation_frac + test_frac),
                                  random_state=random.getstate())
        test, val = test.split(test_frac / (validation_frac + test_frac),
                               random_state=random.getstate())

        random.seed()

        train.sort_key = sort_key
        val.sort_key = sort_key
        test.sort_key = sort_key

        return train, val, test
