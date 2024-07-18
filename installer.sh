#!/usr/bin/env bash


set -e
source $(pwd)/venv/bin/activate

ipython nbconvert deviceCode.ipynb --to python
