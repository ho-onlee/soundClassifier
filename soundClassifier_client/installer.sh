#!/usr/bin/env bash
sudo apt-get update -y 
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
pip install tensorflow==2.16.1 
pip install librosa sounddevice toml pyaudio requests

git clone https://github.com/waveshare/WM8960-Audio-HAT
cd WM8960-Audio-HAT
sudo ./install.sh 
cd ..
sudo rm -rf WM8960-Audio-HAT

if [ -f $(pwd)/soundClassifier_client/config.toml ]; then
  echo 'We have config file!'
fi
read -p "Do you want to build Config? (yN) :" yn
if $yn; then    
    sudo rm config.toml
    echo '###########################'
    echo '#  Building Config file   #'
    echo '###########################'
    echo ''
    # Make Toml.config
    echo '[General]' 
    echo '[General]' >> config.toml
    echo "device_name = '$USER'" 
    echo "device_name = '$USER'" >> config.toml
    echo "client_privilege = 3"
    echo "client_privilege = 3" >> config.toml
    read -p "Enter a node_name: " node_name
    echo "node_name = '$node_name'" 
    echo "node_name = '$node_name'" >> config.toml
    read -p "Enter max thread: " max_thread
    echo "max_thread = "$max_thread
    echo "max_thread = "$max_thread >> config.toml
    echo ''
    echo "">> config.toml
    echo '[Audio_Setting]'
    echo '[Audio_Setting]' >> config.toml
    arecord -l
    read -p "Enter a device (n, n) or none : " device
    echo "device = '$device'" 
    echo "device = '$device'" >> config.toml
    
    read -p "Enter a sample_rate: " sample_rate
    echo 'sample_rate =' $sample_rate  
    echo 'sample_rate =' $sample_rate >> config.toml
    read -p "Enter a duration: " duration
    echo 'duration = '$duration  
    echo 'duration = '$duration >> config.toml
    echo ''
    echo "">> config.toml
    echo '[HOS_server]' 
    echo '[HOS_server]' >> config.toml
    echo 'HOS_available=true' 
    echo 'HOS_available=true' >> config.toml
    echo "server_url = 'hos.seonhunlee.me'" 
    echo "server_url = 'hos.seonhunlee.me'" >> config.toml
    echo 'port =  5051'  
    echo 'port =  5051' >> config.toml
    echo ''
    echo "">> config.toml
    echo '[Weights]' >> config.toml
    echo "model_name='trained_model.h5'" >> config.toml
    echo '[Weights]'  
    echo "model_name='trained_model.h5'" 
    echo ''
    echo "">> config.toml
    echo '[Output]' >> config.toml
    echo '[Output]' 
    read -p "Enter a output_csv_fname: " output_csv_fname
    echo "output_csv_fname = '$output_csv_fname'"  
    echo "output_csv_fname = '$output_csv_fname'" >> config.toml
fi
echo '###################################################################'

# sudo reboot