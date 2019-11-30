FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
MAINTAINER Jangwon Park "adieujw@gmail.com"

# Common steps (must be the same in the CPU and GPU images)
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y vim unzip

### Without this Python thinks we're ASCII and unicode chars fail
ENV LANG C.UTF-8

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python

# Install requirements
RUN pip install sentencepiece>=0.1.82 \
    transformers==2.1.1 \
    gitpython==3.0.2 \
    tensorboard>=1.14.0 \
    tensorboardX==1.8 \
    psutil==5.6.3 \
    scipy==1.3.1 \
    scikit-learn>=0.20.0 \
    seqeval>=0.0.12 \
    gdown

# Erase redundant logs at working directory
RUN rm -i -f *

# Get Repository with data and kobert pretrained model
RUN gdown https://drive.google.com/uc?id=1H-ciRnn54tgCY56OrtsXUyAoObN4Ju6o

# Unzip the file
RUN unzip DistilKoBERT.zip && rm -rf __MACOSX && rm DistilKoBERT.zip