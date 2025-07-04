# Use the official Ubuntu 22.04 image as the base for this container
FROM ubuntu:22.04

# Add system dependencies
RUN apt-get update && \
    apt-get install -y sudo \
        vim \
        make \
        git \
        zip \
        tree \
        curl \
        wget \
        jq \
        software-properties-common \
        python3 \
        python3-pip \
        python3-dev

# Create a user for development
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user with passwordless sudo privileges
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && usermod -aG sudo $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up the Python development environment
WORKDIR /workspaces/mlops-project
RUN python3 -m pip install --upgrade pip wheel

ENV PORT=8000
EXPOSE $PORT

# Enable color terminal for docker exec bash
ENV TERM=xterm-256color

# Become a regular user
USER $USERNAME
