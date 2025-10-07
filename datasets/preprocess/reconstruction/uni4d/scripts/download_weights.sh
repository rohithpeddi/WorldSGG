mkdir -p ./preprocess/pretrained

wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth -O ./preprocess/pretrained/ram_swin_large_14m.pth
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -O ./preprocess/pretrained/scaled_offline.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O ./preprocess/pretrained/sam2.1_hiera_large.pt
wget https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth -O ./preprocess/pretrained/DEVA-propagation.pth