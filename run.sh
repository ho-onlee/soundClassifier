#!/usr/bin bash

git pull
# if [ ! -d $(pwd)/soundClassifier_client/config.toml ]; then
if $1; then
  cd soundClassifier_client
  bash installer.sh
  cd ..
fi
./soundClassifier_client/.venv/bin/python ./soundClassifier_client
