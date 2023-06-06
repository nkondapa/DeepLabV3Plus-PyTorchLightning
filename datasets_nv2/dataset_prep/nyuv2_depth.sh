pip install h5py
pip install gdown

mkdir data/nyu_depth_v2
wget -P data/nyu_depth_v2 http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python datasets/dataset_prep/nyu_v2_depth_extract_official_train_test_set_from_mat.py data/nyu_depth_v2/nyu_depth_v2_labeled.mat data/nyu_depth_v2/splits.mat data/nyu_depth_v2/official_splits/

gdown "1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP" -O data/nyu_depth_v2/  # this will download a zip file named sync.zip
unzip data/nyu_depth_v2/sync.zip -d data/nyu_depth_v2/