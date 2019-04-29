import os

from torchtext.vocab import Vectors

class AugmentedACM(Vectors):
    """
    Word2Vec Embeddings
    ACM (200 dimensions) + Google News (300 dimensions)
    """
    url_base = 'http://wing.comp.nus.edu.sg/acm.500d.txt'

    def __init__(self, **kwargs):
        url = self.url_base
        name = os.path.basename(url)
        super().__init__(name, url=url, **kwargs)
