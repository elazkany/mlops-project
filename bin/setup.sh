#!/bin/bash

set -e  # Exit immediately on error

echo "****************************************"
echo "Setting up MLOps Project Environment"
echo "****************************************"

echo "Installing Python and Virtual Environment"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv docker.io

echo "Checking the Python version..."
python3 --version

#echo "Enabling BuildKit..."
#echo "export DOCKER_BUILDKIT=1" >> ~/.bashrc

if [ ! -f .gitignore ]; then
  touch .gitignore
fi

# Create virtual environment (skip if already exists)
if [ ! -d .venv ]; then
  echo "Creating a Python virtual environment"
  echo ".venv/" >> .gitignore
  python3 -m venv .venv
fi

if [ ! -f .venv/bin/activate ]; then
    echo "Recreating broken or missing virtual environment"
    rm -rf .venv  # Remove partially deleted venv if present
    python3 -m venv .venv
fi

echo "Configure the developer environment"
#echo 'export PS1="\[\e]0;\u:\W\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\$ "' >> ~/.bashrc
echo "source .venv/bin/activate" >> ~/.bashrc

echo "Installing Python dependencies..."
source .venv/bin/activate && python3 -m pip install --upgrade pip wheel
source .venv/bin/activate && pip install -r requirements.txt