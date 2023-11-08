sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop -y gcc zip unzip
mkdir data
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt


# FOR Pascal
wget -P data/ http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf data/VOCtrainval_11-May-2012.tar -C ./data/

rm -rf data/VOCdevkit/VOC2012/SegmentationClass
wget -O data/SegmentationClassAug.zip https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
unzip data/SegmentationClassAug.zip "SegmentationClassAug/*" -d ./data
mv data/SegmentationClassAug data/VOCdevkit/VOC2012/SegmentationClass

mv trainaug.txt data/VOCdevkit/VOC2012/ImageSets/Segmentation/

# For cityscapes
#pip install gdown
#gdown 1x8V3cPQdDKAXP2gxUG3g2APcUXzPG4aM -O data/cityscapes.tar.gz
#tar -xvf data/cityscapes.tar.gz -C data
#mkdir data/cityscapes
#pip install appdirs
#python misc/downloader.py -d ./data/cityscapes gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
#unzip data/cityscapes/gtFine_trainvaltest.zip -d ./data/cityscapes
#unzip data/cityscapes/leftImg8bit_trainvaltest.zip "leftImg8bit/*" -d ./data/cityscapes

