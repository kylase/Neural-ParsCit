import os
import argparse

def init_parser(args):
    DESCRIPTION = 'Neural ParsCit'
    VERSION = '1.0'
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('-m', '--model_path', type=str, help="Directory which the model's parameters are")
    parser.add_argument('-e','--pre_emb', type=argparse.FileType('r'), help="File for word embeddings")
    parser.add_argument('--run', type=str, dest='command', help='Run interactively (shell) or using file (file)')
    parser.add_argument('-i', '--input-file', nargs='?', type=argparse.FileType('r'), help="file containing lines of reference strings")
    parser.add_argument('-o', '--output-file', nargs='?', type=argparse.FileType('w'), help="file to store the output from the model")

    args = parser.parse_args(args)

    if args.command == 'file':
        if not args.input_file:
            raise IOError('Input file is not specified.')

        if not args.output_file:
            raise IOError('Output file is not specified.')
    elif args.command == 'shell':
        pass

    if not args.model_path:
        raise IOError('Model path is not indicated.')
    else:
        MODEL_FILES = set(['word_lstm_rev.mat', 'char_lstm_rev.mat',
                           'mappings.pkl', 'transitions.mat', 'cap_layer.mat',
                           'final_layer.mat','tanh_layer.mat','parameters.pkl',
                           'word_lstm_for.mat', 'char_lstm_for.mat',
                           'char_layer.mat'])
        if not set(os.listdir(args.model_path)) <= MODEL_FILES:
            raise Exception

    return parser
