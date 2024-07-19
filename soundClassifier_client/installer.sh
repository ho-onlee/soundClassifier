#!/usr/bin/env bash

sudo apt-get install python3-pyaudio -y
sudo apt-get install libasound-dev -y
sudo apt install portaudio19-dev -y
sudo apt-get install libasound2-dev -y
sudo apt-get install libsndfile-dev -y


python3 -m venv .venv
source $(pwd)/.venv/bin/activate

pip install HOS-client -i https://pip.seonhunlee.me/simple
pip install tensorflow librosa sounddevice toml pyaudio

