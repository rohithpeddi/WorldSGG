#!/usr/bin/env python3
"""
Example script to run GroundingDINO zero-shot evaluation on Action Genome dataset
"""

import subprocess
import sys
import os

def run_evaluation():
    """
    Example function to run the GroundingDINO evaluation
    """
    
    # Configuration - Update these paths according to your setup
    ag_root_directory = "/data/rohith/ag/"  # Update this path
    gdino_predictions_path = None  # Will use default: ag_root_directory/detection/gdino
    output_file = "gdino_evaluation_results.json"
    
    # Check if the Action Genome directory exists
    if not os.path.exists(ag_root_directory):
        print(f"Error: Action Genome directory not found at {ag_root_directory}")
        print("Please update the ag_root_directory path in this script")
        return
    
    # Construct the command
    cmd = [
        sys.executable, "test_gdino_zero_shot.py",
        "--ag_root_directory", ag_root_directory,
        "--output_file", output_file
    ]
    
    if gdino_predictions_path:
        cmd.extend(["--gdino_predictions_path", gdino_predictions_path])
    
    print("Running GroundingDINO evaluation...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("-" * 60)
        print("Evaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install pycocotools torch numpy tqdm")
        
    except FileNotFoundError:
        print("Error: test_gdino_zero_shot.py not found in current directory")

def install_dependencies():
    """
    Install required dependencies
    """
    dependencies = ["pycocotools", "torch", "numpy", "tqdm"]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}")
            return False
    
    print("All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    print("GroundingDINO Zero-Shot Evaluation Example")
    print("=" * 50)
    
    # Option 1: Install dependencies first (uncomment if needed)
    # print("Installing dependencies...")
    # if not install_dependencies():
    #     sys.exit(1)
    
    # Option 2: Run evaluation directly
    run_evaluation()
