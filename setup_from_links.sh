sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop -y gcc zip unzip

git clone https://github.com/nkondapa/DeeplabV3PlusPL.git
cd DeeplabV3PlusPL
mkdir data

pip install -r requirements.txt

wget -P data/ http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf data/VOCtrainval_11-May-2012.tar -C ./data/

rm -rf data/VOCdevkit/VOC2012/SegmentationClass
wget -O data/SegmentationClassAug.zip https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
unzip data/SegmentationClassAug.zip "SegmentationClassAug/*" -d ./data
mv data/SegmentationClassAug data/VOCdevkit/VOC2012/SegmentationClass

mv trainaug.txt data/VOCdevkit/VOC2012/ImageSets/Segmentation/