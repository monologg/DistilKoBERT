FROM ubuntu:18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

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