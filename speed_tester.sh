#!/usr/bin bash

git pull
cd ./soundClassifier_client
if [ ! -d ".env" ]; then
  python3 -m venv .env
fi

# Must install for sounddevice to work
sudo apt-get install libportaudio2

source ./soundClassifier_client/.env/bin/activate
pip install tensorflow==2.16.2 numpy scipy librosa sounddevice
.env/bin/python ./speed_tester.py
j