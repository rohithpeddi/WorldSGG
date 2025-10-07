# -------------------------- BONN --------------------------

WORKDIR="./data/bonn"

echo "Starting preprocessing for Bonn"
    
# RAM-GPT step
echo "Starting RAM-GPT"
python ./preprocess/ram_gpt.py --workdir "$WORKDIR"
python ./preprocess/run_cotracker.py --workdir "$WORKDIR" --interval 10 --grid_size 50
python ./preprocess/run_unidepth.py --workdir "$WORKDIR"
python ./preprocess/run_unidepth.py --workdir "$WORKDIR" --use_gt_K

# Grounded-SAM-2 step
echo "Starting DINO + SAM2"
python ./preprocess/run_dino_sam2.py --workdir "$WORKDIR"

# DEVA step
echo "Starting DEVA"
python Tracking-Anything-with-DEVA/evaluation/eval_with_detections.py --workdir "$WORKDIR" --model ./preprocess/pretrained/DEVA-propagation.pth

# # -------------------------- SINTEL --------------------------

WORKDIR="./data/sintel"

echo "Starting preprocessing for Sintel"
    
# RAM-GPT step
echo "Starting RAM-GPT"
python ./preprocess/ram_gpt.py --workdir "$WORKDIR"
python ./preprocess/run_cotracker.py --workdir "$WORKDIR" --interval 10 --grid_size 75
python ./preprocess/run_unidepth.py --workdir "$WORKDIR"
python ./preprocess/run_unidepth.py --workdir "$WORKDIR" --use_gt_K

# Grounded-SAM-2 step
echo "Starting DINO + SAM2"
python ./preprocess/run_dino_sam2.py --workdir "$WORKDIR"

# DEVA step
echo "Starting DEVA"
python Tracking-Anything-with-DEVA/evaluation/eval_with_detections.py --workdir "$WORKDIR" --model ./preprocess/pretrained/DEVA-propagation.pth

# # -------------------------- TUMD --------------------------

WORKDIR="./data/tumd"

echo "Starting preprocessing for TUMD"
    
# RAM-GPT step
echo "Starting RAM-GPT"
python ./preprocess/ram_gpt.py --workdir "$WORKDIR"
python ./preprocess/run_cotracker.py --workdir "$WORKDIR" --interval 10 --grid_size 50
python ./preprocess/run_unidepth.py --workdir "$WORKDIR"
python ./preprocess/run_unidepth.py --workdir "$WORKDIR" --use_gt_K

# Grounded-SAM-2 step
echo "Starting DINO + SAM2"
python ./preprocess/run_dino_sam2.py --workdir "$WORKDIR"

# DEVA step
echo "Starting DEVA"
python Tracking-Anything-with-DEVA/evaluation/eval_with_detections.py --workdir "$WORKDIR" --model ./preprocess/pretrained/DEVA-propagation.pth

# # -------------------------- KITTI --------------------------

WORKDIR="./data/kitti"

echo "Starting preprocessing for KITTI"
    
# RAM-GPT step
echo "Starting RAM-GPT"
python ./preprocess/ram_gpt.py --workdir "$WORKDIR"
python ./preprocess/run_cotracker.py --workdir "$WORKDIR" --interval 10 --grid_size 50
python ./preprocess/run_unidepth.py --workdir "$WORKDIR"

# Grounded-SAM-2 step
echo "Starting DINO + SAM2"
python ./preprocess/run_dino_sam2.py --workdir "$WORKDIR"

# DEVA step
echo "Starting DEVA"
python Tracking-Anything-with-DEVA/evaluation/eval_with_detections.py --workdir "$WORKDIR" --model ./preprocess/pretrained/DEVA-propagation.pth
