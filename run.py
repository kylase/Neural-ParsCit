from __future__ import print_function
import os
import re
import optparse
import json
import numpy as np
import theano
from contextlib import closing
from utils import evaluate, create_input
from model import Model
from loader import augment_with_pretrained, load_sentences, prepare_dataset


optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model_path", default="",
    help="Model location"
)
optparser.add_option(
    "-e", "--pre_emb", default="",
    help="Pretrained embeddings location"
)
optparser.add_option(
    "-r", "--run", default="shell",
    help="Run interactively (=shell) or using file (=file)"
)
optparser.add_option(
    "-i", "--input_file", default="0",
    help="location of input file"
)
optparser.add_option(
    "-o", "--output_file", default="0",
    help="location of output file"
)
opts = optparser.parse_args()[0]

model = Model(model_path=opts.model_path)
model.parameters['pre_emb'] = os.path.join(os.getcwd(), opts.pre_emb)
f = model.build(training=False, **model.parameters)

model.reload()

lower = model.parameters['lower']
zeros = model.parameters['zeros']

word_to_id = {v:i for i, v in model.id_to_word.items()}
char_to_id = {v:i for i, v in model.id_to_char.items()}

while True:
    if opts.run == 'file':
        assert opts.input_file
        assert opts.output_file

        output_file = opts.output_file

        with closing(open(opts.input_file, 'r')) as fh:
            data = fh.read()
        strings = data.split('\n')
    else:
        string = raw_input("Enter the citation string: ")
        strings = [string]

    test_file = "test_file"
    if os.path.exists(test_file):
        os.remove(test_file)
    file = open(test_file, 'a')
    for string in strings:
        file.write('\n'.join(string.split()) + '\n')
    file.close()
    test_sentences = load_sentences(test_file, lower, zeros)
    data = prepare_dataset(test_sentences, word_to_id, char_to_id, {}, lower, True)

    for citation in data:
        inputs = create_input(citation, model.parameters, False)
        y_pred = np.array(f[1](*inputs))[1:-1]

        tags = [model.id_to_tag[y_pred[i]] for i in range(len(y_pred))]

        output = [w + '\t' + tags[i] for i, w in enumerate(citation['str_words'])]

        if opts.run == 'file':
            with closing(open(output_file, 'w')) as fh:
                fh.write('\n'.join(output))
        else:
            print('\n'.join(output))

    if opts.run == 'file':
        break
