#!/usr/bin/env bash

sudo apt-get install python3-pyaudio -y
sudo apt-get install libasound-dev -y
sudo apt install portaudio19-dev -y
sudo apt install pkg-config libhdf5-dev libsndfile-dev libasound2-dev -y

if [ ! -d $(pwd)/.venv ]; then
  python3 -m venv .venv
  sudo chown -R $USER:$USER $(pwd)/.venv
fi
source $(pwd)/.venv/bin/activate

pip install HOS-client -i https://pip.seonhunlee.me/simple
pip install tensorflow librosa sounddevice toml pyaudio

git clone https://github.com/waveshare/WM8960-Audio-HAT
cd WM8960-Audio-HAT
sudo ./install.sh 
# sudo rm -rf seeed-voicecard

cd ..
# sudo reboot