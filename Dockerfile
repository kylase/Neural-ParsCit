FROM python:2

WORKDIR /usr/src

RUN apt-get update \
  && apt-get install -y libopenblas-dev \
  && apt-get clean

RUN pip install --no-cache-dir Theano==0.10.0beta4 numpy==1.13.3 gensim==0.13.2
