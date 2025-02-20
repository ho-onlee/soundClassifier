#!/usr/bin bash

git pull
cd ./soundClassifier_client
if [ ! -d ".env" ]; then
  python3 -m venv .env
fi

source ./soundClassifier_client/.env/bin/activate
pip install tensorflow numpy scipy librosa sounddevice

.env/bin/python ./speed_tester.py
