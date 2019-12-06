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
    python3-pip \
    python3-dev\
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

### Without this Python thinks we're ASCII and unicode chars fail
ENV LANG C.UTF-8

# Working directory
COPY . /app
WORKDIR /app

# Install requirements
RUN pip3 install -r requirements.txt --no-cache-dir

# Save pretrained model in advance
RUN python3 download_models.py

# Run the server
EXPOSE 80
CMD [ "python3", "app.py", "--model_type", "nsmc", "--add_special_tokens", "--no_cuda", "--port_num", "80"]