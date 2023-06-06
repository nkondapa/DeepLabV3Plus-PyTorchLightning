sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop -y gcc
git clone https://github.com/nkondapa/DeeplabV3PlusPL.git
cd DeeplabV3PlusPL
mkdir data
sudo apt-get install zip unzip
cd ..
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt