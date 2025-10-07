WORKDIR="./data/demo"

echo "Starting Demo"
    
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

# Uni4D step
echo "Starting Uni4D optimization"
python ./uni4d/run.py --config ./uni4d/config/config_demo.yaml

# Uni4D Visualization
echo "Starting Visualization"
python ./uni4d/vis_final_4d.py --results_path ./data/demo/lady-running/uni4d/demo/fused_4d.npz

