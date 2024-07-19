#!/usr/bin/env bash

python3 -m venv .venv
source $(pwd)/.venv/bin/activate

pip install HOS-client -i https://pip.seonhunlee.me/simple
pip install tensorflow librosa sounddevice toml

