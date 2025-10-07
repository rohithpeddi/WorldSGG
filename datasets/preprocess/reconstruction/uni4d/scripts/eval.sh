python uni4d/eval_pose.py --dataset tumd --base_path ./data/tumd --experiment_name base
python uni4d/eval_pose.py --dataset sintel --base_path ./data/sintel --experiment_name base
python uni4d/eval_pose.py --dataset bonn --base_path ./data/bonn --experiment_name base

python uni4d/eval_pose.py --dataset tumd --base_path ./data/tumd --experiment_name base_gt_K
python uni4d/eval_pose.py --dataset sintel --base_path ./data/sintel --experiment_name base_gt_K
python uni4d/eval_pose.py --dataset bonn --base_path ./data/bonn --experiment_name base_gt_K

python uni4d/eval_depth.py --dataset kitti --base_path ./data/kitti --experiment_name base
python uni4d/eval_depth.py --dataset sintel --base_path ./data/sintel --experiment_name base
python uni4d/eval_depth.py --dataset bonn --base_path ./data/bonn --experiment_name base