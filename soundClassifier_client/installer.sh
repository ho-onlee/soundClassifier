#!/usr/bin/env bash

if $(pwd)

mkdir soundClassifier
cd soundClassifier
git pull https://github.com/ho-onlee/soundClassifier.git

pip install 
set -e
source $(pwd)/venv/bin/activate

ipython nbconvert deviceCode.ipynb --to python
