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

# Working directory
COPY ../server /app
WORKDIR /app

# Install requirements
RUN pip3 install -r requirements

# Run the server
ENTRYPOINT ["python3"]
CMD ["app.py --model_type nsmc --add_special_tokens --no_cuda"]