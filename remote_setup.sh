sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop -y gcc zip unzip
git clone https://github.com/nkondapa/DeeplabV3PlusPL.git
cd DeeplabV3PlusPL
mkdir data
cd ..
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt