
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root for package imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.detector.monocular3d.datasets.ag_dataset_3d import ActionGenomeDataset3D, collate_fn
from lib.detector.monocular3d.models.dino_mono_3d import DinoV2Monocular3D

def test_dataset():
    print("Testing Dataset...")
    data_path = "/data/rohith/ag/"
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist. Skipping dataset test with real data.")
        return None

    dataset = ActionGenomeDataset3D(data_path, phase='train', target_size=224)
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Target keys: {target.keys()}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Boxes 3D shape: {target['boxes_3d'].shape}")
        
        assert target['boxes_3d'].shape[1:] == (8, 3), f"Expected (N, 8, 3), got {target['boxes_3d'].shape}"
        return dataset
    return None

def test_model(dataset=None):
    print("\nTesting Model...")
    model = DinoV2Monocular3D(num_classes=37, pretrained=False, model="v3l") # Use pretrained=False for speed
    model.train()
    
    # Create dummy input if dataset not available
    if dataset is None:
        print("Creating dummy input...")
        images = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]
        targets = [
            {
                'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32),
                'labels': torch.tensor([1, 2], dtype=torch.int64),
                'boxes_3d': torch.randn(2, 8, 3, dtype=torch.float32)
            },
            {
                'boxes': torch.tensor([[20, 20, 80, 80]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'boxes_3d': torch.randn(1, 8, 3, dtype=torch.float32)
            }
        ]
    else:
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        images, targets = next(iter(loader))
        
    print("Forward pass...")
    losses = model(images, targets)
    print("Losses:", losses)
    
    if 'loss_3d' in losses:
        print("loss_3d present.")
    else:
        print("Error: loss_3d missing!")
        
    total_loss = sum(losses.values())
    print("Backward pass...")
    total_loss.backward()
    print("Backward pass successful.")

if __name__ == "__main__":
    dataset = test_dataset()
    test_model(dataset)
