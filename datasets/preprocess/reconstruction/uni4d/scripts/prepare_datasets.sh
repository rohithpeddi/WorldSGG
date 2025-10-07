cd dataset_prepare

bash download_bonn.sh
bash download_kitti.sh
bash download_sintel.sh
bash download_tum.sh

cd ..

python ./dataset_prepeare/preprocess_bonn.py
python ./dataset_prepare/preprocess_kitti.py
python ./dataset_prepare/preprocess_sintel.py
python ./dataset_prepare/preprocess_tumd.py
