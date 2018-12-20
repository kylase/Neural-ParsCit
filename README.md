## Neural ParsCit

This is the official repository of Neural-ParsCit and is under active development at National University of Singapore (NUS), Singapore

[![Build Status](https://travis-ci.com/WING-NUS/Neural-ParsCit.svg?branch=master)](https://travis-ci.com/WING-NUS/Neural-ParsCit)

Neural ParsCit is a citation string parser which parses reference strings into its component tags such as Author, Journal, Location, Date, etc. Neural ParsCit uses Long Short Term Memory (LSTM), a deep learning model to parse the reference strings. This deep learning algorithm is chosen as it is designed to perform sequence-to-sequence labeling tasks such as ours. Input to the model are word embeddings which are vector representation of words. We provide word embeddings as well as character embeddings as input to the network.


## Initial setup

To use the tagger, you need Python 2.7 (works in Python 3 but not fully supported), with Numpy, Theano and Gensim installed. scikit-learn is needed for model evaluation if you are training a new model.

You can use environmental variables to set the following:
- `MODEL_PATH`: Path to the model's parameters
- `WB_PATH`: Path to the word embeddings
- `TIMEOUT`: Timeout for gunicorn when starting the Flask app. Increase this if you experience the Flask app is unable to start as the model building process takes too long. [Default: 60]
- `NUM_WORKERS`: Number of workers which gunicorn spawns. [Default: 1]

### Using virtualenv in Linux systems

```
virtualenv -ppython2.7 .venv
source .venv/bin/activate
pip install -r requirements/<env>.txt
```

Where `<env>` is `{prod, dev, test}`

### Using Docker

1. Build the image: `docker build -t theano-gensim - < Dockerfile`
1. Run the repo mounted to the container: `docker run -it -v $(pwd):/usr/src --name np theano-gensim:latest /bin/bash`

## Word Embeddings

The word embeddings do not come with this repository. You can obtain the [word embeddings with `<UNK>`](http://wing.comp.nus.edu.sg/~wing.nus/resources/NParsCit/vectors_with_unk.tar.gz) from WING website. Please read the next section on availability of `<UNK>` in word embeddings.

You will need to extract the content of the word embedding archive (`vectors_with_unk.tar.gz`) to the root directory for this repository by running `tar xfz vectors_with_unk.tar.gz`.

### Embeddings Without `<UNK>`

If the word embeddings provided do not have `<UNK>`, your instance will not benefit from the lazy loading of the word vectors and hence the reduction of memory requirements.

Without `<UNK>`, at most 7.5 GB of memory is required as the entire word vectors need to be instantiated in memory to create the new matrix. Comparing with embeddings with `<UNK>`, which is much lower as it only requires at most 4.5 GB.


## Parse citation strings

### Command Line

The fastest way to use the parser is to run state-of-the-art pre-trained model as follows:

```
./run.py --model_path models/neuralParsCit/ --pre_emb <vectors.bin> --run shell
./run.py --model_path models/neuralParsCit/ --pre_emb <vectors.bin> --run file -i input_file -o output_file
```
The script can run interactively or input can be passed in a file. In the interactive session, the strings are passed one by one. The result is displayed on standard output. If the file option is chosen, the input is given in a file specified by -i option and the output is stored in the directed file. Using the file option, multiple citation strings can be parsed.

The state-of-the-art trained model is provided in the models folder and is named neuralParsCit. The binary file for word embeddings is provided in the docker image of the current version of neural ParsCit. The hyper parameter ```discarded``` is the number of embeddings not used in our model. Retained words have a frequency of more than 0 in the ACM citation literature from 1994-2014.

### Using a Web Server

Note: This service is not Python 3 compatible due to `unicode`.

The web server (a Flask app) provides REST API.

Running the web server,
`docker run --rm -it -p 8000:8000 -e TIMEOUT=60 -v $(pwd):/usr/src --name np theano-gensim:latest /bin/bash`

In the container, `gunicorn -b 0.0.0.0:8000 -w $NUM_WORKERS --timeout $TIMEOUT run_app:app`

The REST API documentation can be found at `http://localhost:8000/docs`


## Train a model

To train your own model, you need to use the train.py script and provide the location of the training, development and testing set:

```
./train.py --train train.txt --dev dev.txt --test test.txt
```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run:

```
./train.py --help
```

Input files for the training script have to follow the following format: each word of the citation string and its corresponding tag has to be on a separate line. All citation strings must be separated by a blank line.

Details about the training data, experiments can be found in the following article. Training data and CRF baseline can be downloaded from https://github.com/knmnyn/ParsCit. Please consider citing following publication(s) if you use Neural ParsCit:
```
@article{animesh2018neuralparscit,
  title={Neural ParsCit: A Deep Learning Based Reference String Parser},
  author={Prasad, Animesh and Kaur, Manpreet and Kan, Min-Yen},
  journal={International Journal on Digital Libraries},
  volume={19},
  pages={323-337},
  year={2018},
  publisher={Springer},
  url={https://link.springer.com/article/10.1007/s00799-018-0242-1}
}
```
