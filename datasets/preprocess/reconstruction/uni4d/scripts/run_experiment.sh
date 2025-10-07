# -------------------------- BONN --------------------------

# Uni4D step
echo "Starting Uni4D optimization on Bonn"

python ./uni4d/run.py --config ./uni4d/config/config_experiment.yaml --experiment_name base --workdir ./data/bonn/ --opt_intrinsics
python ./uni4d/run.py --config ./uni4d/config/config_experiment.yaml  --experiment_name base_gt_K --depth_dir unidepth_gt_K --workdir ./data/bonn/ 

# # -------------------------- SINTEL --------------------------
# Uni4D step
echo "Starting Uni4D optimization on Sintel"

python ./uni4d/run.py --config ./uni4d/config/config_experiment.yaml --experiment_name base --opt_intrinsics --workdir ./data/sintel/ --cotracker_path cotrackerv3_10_75
python ./uni4d/run.py --config ./uni4d/config/config_experiment.yaml  --experiment_name base_gt_K --depth_dir unidepth_gt_K --workdir ./data/sintel/ --cotracker_path cotrackerv3_10_75 

# # -------------------------- TUMD --------------------------

# Uni4D step
echo "Starting Uni4D optimization on TUMD"

python ./uni4d/run.py --config ./uni4d/config/config_experiment.yaml --experiment_name base --workdir ./data/tumd/ --opt_intrinsics
python ./uni4d/run.py --config ./uni4d/config/config_experiment.yaml  --experiment_name base_gt_K --workdir ./data/tumd/ --depth_dir unidepth_gt_K 

# # -------------------------- KITTI --------------------------

# Uni4D step
echo "Starting Uni4D optimization on KITTI"

python ./uni4d/run.py --config ./uni4d/config/config_experiment.yaml --experiment_name base --workdir ./data/kitti/ --opt_intrinsics
