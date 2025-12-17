import argparse
import os
import warnings

import numpy as np
import torch
import wandb
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
# from dataloader.ag_dataset import ActionGenomeDataset,collate_fn
# from dataloader.ag_dataset_letterbox import ActionGenomeDatasetLetterbox as ActionGenomeDataset, collate_fn
from dataloader.ag_dataset_resize import ActionGenomeDatasetResize as ActionGenomeDataset, collate_fn
from model.dinov2_torch import create_model

from evaluate import evaluate_MAP_full
from torch.utils.data import Subset
from torch_utils import utils
#from torch_utils.coco_utils import get_coco_api_from_dataset
from utils.json_logger import LocalLogger
from accelerate import Accelerator

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from huggingface_hub import login
login("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")

def plot_predictions_after_epoch(model, dataset, device, epoch, sample_idx=12, save_dir=None, score_threshold=0.05):
    """
    Plot predictions on a sample image after each epoch.
    
    Args:
        model: Trained model (in eval mode)
        dataset: Test dataset
        device: Device to run inference on
        epoch: Current epoch number
        sample_idx: Index of sample to visualize
        save_dir: Directory to save visualization
        score_threshold: Minimum confidence score to show
    """
    if save_dir is None:
        return
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Get sample
    if sample_idx >= len(dataset):
        sample_idx = 0
    
    image_tensor, target = dataset[sample_idx]
    sample = dataset.samples[sample_idx]
    
    # Get normalization params
    mean = tuple(dataset.image_mean)
    std = tuple(dataset.image_std)
    
    # Run inference
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        predictions = model(image_batch)
    
    pred = predictions[0]
    pred_boxes = pred['boxes'].detach().cpu().numpy()
    pred_labels = pred['labels'].detach().cpu().numpy()
    pred_scores = pred['scores'].detach().cpu().numpy()
    
    # Filter by score threshold
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    
    # Get GT boxes
    gt_boxes = target['boxes'].detach().cpu().numpy() if isinstance(target['boxes'], torch.Tensor) else target['boxes']
    gt_labels = target['labels'].detach().cpu().numpy() if isinstance(target['labels'], torch.Tensor) else target['labels']
    
    # Convert image to numpy
    def tensor_to_image_np(img_tensor, mean, std):
        img = img_tensor.detach().cpu().float().clone()
        mean_t = torch.tensor(mean, dtype=img.dtype).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=img.dtype).view(3, 1, 1)
        img = img * std_t + mean_t
        img = img.clamp(0.0, 1.0)
        img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return img
    
    image_np = tensor_to_image_np(image_tensor, mean, std)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image_np)
    ax.set_title(f"Epoch {epoch+1} | Sample: {sample['filename']}\n"
                 f"GT: {len(gt_boxes)} boxes | Pred: {len(pred_boxes)} boxes (score>={score_threshold})")
    ax.axis('off')
    
    # Draw GT boxes (cyan)
    for box, lab in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box.tolist() if isinstance(box, (torch.Tensor, np.ndarray)) else box
        color = 'cyan'
        rect = patches.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1), 
                                 linewidth=2.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        class_name = dataset.object_classes[int(lab)] if 0 <= int(lab) < len(dataset.object_classes) else str(lab)
        ax.text(max(0, x1), max(10, y1 - 2), f"{class_name}", fontsize=8, color='black',
                verticalalignment='top', bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Draw predicted boxes (lime) with scores
    for box, lab, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box.tolist() if isinstance(box, (torch.Tensor, np.ndarray)) else box
        color = 'lime'
        rect = patches.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1),
                                 linewidth=2.0, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        class_name = dataset.object_classes[int(lab)] if 0 <= int(lab) < len(dataset.object_classes) else str(lab)
        label_text = f"{class_name} ({score:.2f})"
        ax.text(max(0, x1), max(10, y1 - 2), label_text, fontsize=8, color='black',
                verticalalignment='top', bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=3, label='Ground Truth'),
        Line2D([0], [0], color='lime', lw=3, label=f'Predictions (score>={score_threshold})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save
    save_path = os.path.join(save_dir, f"predictions_epoch_letterbox_{epoch+1:03d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def configure_rpn_and_roi(model):
    rpn = getattr(model, 'rpn', None)
    if rpn is not None:
        # Newer torchvision API (preferred)
        if hasattr(rpn, 'pre_nms_top_n_train') and hasattr(rpn, 'post_nms_top_n_train'):
            rpn.pre_nms_top_n_train  = 2000    # was 1000
            rpn.post_nms_top_n_train = 1000   # was 256
            rpn.pre_nms_top_n_test   = 2000
            rpn.post_nms_top_n_test  = 500
        # Older API (dict-based), only if attribute is actually a dict
        elif hasattr(rpn, 'pre_nms_top_n') and isinstance(rpn.pre_nms_top_n, dict):
            rpn.pre_nms_top_n  = {'train': 2000, 'test': 2000}
            rpn.post_nms_top_n = {'train': 1000, 'test': 500}

        # Lower RPN loss sampler cost
        if hasattr(rpn, 'batch_size_per_image'):
            rpn.batch_size_per_image = 128   # default 256
        if hasattr(rpn, 'positive_fraction'):
            rpn.positive_fraction = 0.5     # keep

    roi_heads = getattr(model, 'roi_heads', None)
    if roi_heads is not None:
        if hasattr(roi_heads, 'batch_size_per_image'):
            roi_heads.batch_size_per_image = 256   # was 128
        if hasattr(roi_heads, 'detections_per_img'):
            roi_heads.detections_per_img = 200     # was 100

    # Downscale detector input a bit (big speed win if your images are large)
    if hasattr(model, 'transform'):
        # tuple ensures torchvision treats it as fixed min size
        model.transform.min_size = (512,)
        model.transform.max_size = 1024



parser = argparse.ArgumentParser(description="Train DINOv2 AG object detector")
parser.add_argument("--experiment_name",help="Name of the experiment being launched", required=True, type=str)
parser.add_argument("--working_dir", help="working directory where model, logs are saved", default = "/home/cse/msr/csy227518/scratch/Projects/Active/Practice/ag_object_detection/train_data", type=str)
parser.add_argument("--use_collate",action='store_true')
parser.add_argument("--use_wandb",action='store_true')
parser.add_argument("--no_collate",action='store_false',dest="use_collate")
parser.add_argument("--no_wandb",action='store_false', dest="use_wandb")
parser.add_argument("--lr",type=float,default=1e-4)  # Higher LR for better mAP convergence
parser.add_argument("--batch_size",type=int,default=64)  # Increased from 4
parser.add_argument("--epochs",type=int,default=70)
parser.add_argument("--data_path",type=str,default='/home/cse/msr/csy227518/scratch/Datasets/action_genome')
parser.add_argument("--save_path",type=str,default='/home/cse/msr/csy227518/scratch/Projects/Active/Practice/ag_object_detection/save_models')
parser.add_argument("--ckpt",type=str,default=None)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_grad_norm", help = "Maximum norm for gradient clipping", default=1.0, type=float)
parser.set_defaults(use_wandb=True)
parser.set_defaults(use_collate=True)
args = parser.parse_args()


## Init Accelerator ##
path_to_experiment = os.path.join(args.working_dir, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                        gradient_accumulation_steps = args.gradient_accumulation_steps,
                        log_with = "wandb")
                        # mixed_precision = "fp16")

## Init Logger ##

local_logger = LocalLogger(path_to_experiment)

## Wndb Logger ##
experiment_config = {"epochs" : args.epochs,
                    "Effective_batch_size": args.batch_size*accelerator.num_processes,
                    "learning_rate":args.lr}
accelerator.init_trackers(args.experiment_name, config=experiment_config)

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.use_wandb:
    wandb.init(
        project="DINOv2-Object-Detector-AG",
        config={
            "learning_rate":args.lr,
            "epochs":args.epochs,
            "batch_size":args.batch_size,
            "dataset":"Action_Genome"
        }
    )

config = wandb.config

# train_dataset = ActionGenomeDataset(data_path = args.data_path, phase = "train")
# test_dataset = ActionGenomeDataset(data_path = args.data_path,phase = "test")

train_dataset = ActionGenomeDataset(data_path=args.data_path, phase="train", target_size=1024)
test_dataset = ActionGenomeDataset(data_path=args.data_path, phase="test", target_size=1024)

subset_ind = list(range(2000))
subset_ind_2 = list(range(180000))
test_dataset_subset = Subset(test_dataset, subset_ind)
train_dataset_subset = Subset(train_dataset, subset_ind_2)

train_dataloader = DataLoader(train_dataset_subset, batch_size = args.batch_size, shuffle = True, num_workers = 16, collate_fn = collate_fn, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 16, collate_fn = collate_fn, drop_last=True, pin_memory=True)
test_dataloader_subset = DataLoader(test_dataset_subset, batch_size = args.batch_size, shuffle = False, num_workers = 32, pin_memory=True, collate_fn = collate_fn)  

NUM_CLASSES = len(train_dataset.object_classes)
model = create_model(num_classes = NUM_CLASSES,pretrained = True, use_fpn = True, model="v3l")
configure_rpn_and_roi(model)

total_steps = args.epochs*len(train_dataloader) 
warmup_steps = int(0.01 * total_steps)
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
# replace optimizer init
head_params = [p for n, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=0.001)
# optimizer = SGD(head_params, lr=1e-3, momentum=0.9, nesterov=True)

warmup = LinearLR(optimizer, start_factor=1e-1, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=args.lr * 0.1)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

## Prepare Everything for acceleratoe ##
model, optimizer, train_dataloader, test_dataloader, test_dataloader_subset, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader, test_dataloader_subset, scheduler
)


accelerator.register_for_checkpointing(scheduler)

## Check if ckpt is passed or not ##
if args.ckpt is not None:
    accelerator.print(f"Resuming from checkpoint : {args.ckpt}")
    path_to_checkpoint = os.path.join(path_to_experiment, args.ckpt)
    checkpoint_file = os.path.join(path_to_checkpoint, 'checkpoint_state.pth')
    
    if os.path.exists(checkpoint_file):
        # Load checkpoint with model, optimizer, scheduler, epoch, and LR
        # Use CPU first, then move to device after model_device is defined
        checkpoint_state = torch.load(checkpoint_file, map_location='cpu')       
        # Load model state
        accelerator.unwrap_model(model).load_state_dict(checkpoint_state['model_state_dict'])
        # Load optimizer state
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        # Load scheduler state
        scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])
        # Get epoch to resume from
        starting_checkpoint = checkpoint_state.get('epoch', 0) + 1
        
        accelerator.print(f"✓ Loaded checkpoint from epoch {checkpoint_state.get('epoch', 0) + 1}")
        accelerator.print(f"  Resuming from epoch {starting_checkpoint}")
        accelerator.print(f"  Checkpoint LR: {checkpoint_state.get('current_lr', 'N/A')}")
    else:
        accelerator.print(f"⚠️  Checkpoint file not found at {checkpoint_file}")
        starting_checkpoint = int(args.ckpt.split("_")[-1]) if "_" in args.ckpt else 0
else:
    starting_checkpoint = 0

## Train_loop ##

model_device = next(model.parameters()).device
model = model.to(model_device, memory_format=torch.channels_last)


# Runtime speed toggles
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    pass

# configure_rpn_and_roi(model)

for epoch in range(starting_checkpoint, args.epochs):
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []
    train_loss_hist = []

    epoch_loss_cls_list = []
    epoch_loss_box_reg_list = []
    epoch_loss_objectness_list = []
    epoch_loss_rpn_list = []

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_box_loss = 0.0
    running_object_loss = 0.0
    running_rpn_loss = 0.0
    running_count = 0

    best_mAP = 0.0
    global_iteration = 0
    box_weight = 1.0
    model.train()

    for images, targets in tqdm(train_dataloader,ascii=True):
        images = torch.stack([img for img in images]).to(model_device, non_blocking=True)
        images = images.contiguous(memory_format=torch.channels_last)
        if args.use_collate:
            targets = [
                {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
        else:
            targets = [{k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}]

        with accelerator.accumulate(model):
            ## Pass through model ##
            
            # start = time.time()
            # backbone = model.backbone.to(model_device)
            # x = backbone(images)
            # end = time.time()

            # print(f" ########### \n  BACKBONE TIME : {end-start}\n #############")
            # start = time.time()
            
            with accelerator.autocast():
                loss_dict_original = model(images, targets)
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #     loss_dict_original = model(images, targets)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            # end = time.time()
            # print(f" ########### \n  FASTERRCNN TIME : {end-start}\n #############")
            loss_dict = {}
            for k, v in loss_dict_original.items():
                if k in ("loss_box_reg", "loss_rpn_box_reg"):
                    loss_dict[k] = box_weight * v
                else:
                    loss_dict[k] = v
            ## Calculate Losses ##
            losses = sum(loss/args.gradient_accumulation_steps for loss in loss_dict.values())
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss/args.gradient_accumulation_steps  for loss in loss_dict_reduced.values())

            ## Calculate Gradients ##
            accelerator.backward(losses)

            ## Clip Gradients ##
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if accelerator.is_main_process:
                batch_loss_list.append(losses_reduced.item())
                # component-wise
                batch_loss_cls_list.append(loss_dict_reduced.get('loss_classifier', torch.tensor(0.0)).item())
                batch_loss_box_reg_list.append(loss_dict_reduced.get('loss_box_reg', torch.tensor(0.0)).item())
                batch_loss_objectness_list.append(loss_dict_reduced.get('loss_objectness', torch.tensor(0.0)).item())
                batch_loss_rpn_list.append(loss_dict_reduced.get('loss_rpn_box_reg', torch.tensor(0.0)).item())

                # Update running averages
                running_total_loss += losses_reduced.item()
                running_cls_loss += loss_dict_reduced.get('loss_classifier', torch.tensor(0.0)).item()
                running_box_loss += loss_dict_reduced.get('loss_box_reg', torch.tensor(0.0)).item()
                running_object_loss += loss_dict_reduced.get('loss_objectness', torch.tensor(0.0)).item()
                running_rpn_loss += loss_dict_reduced.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
                running_count += 1
            ## Update Model Parameters ##
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            scheduler.step()

            global_iteration += 1


            # Log losses every 25 iterations
            if global_iteration % 10000 == 0 and accelerator.is_main_process:
                avg_total_loss = running_total_loss / running_count
                avg_cls_loss = running_cls_loss / running_count
                avg_box_loss = running_box_loss / running_count
                avg_object_loss = running_object_loss / running_count
                avg_rpn_loss = running_rpn_loss / running_count
                
                if args.use_wandb:
                    wandb.log({
                        "iteration": global_iteration,
                        "iter/total_loss": avg_total_loss,
                        "iter/cls_loss": avg_cls_loss,
                        "iter/box_loss": avg_box_loss,
                        "iter/object_loss": avg_object_loss,
                        "iter/rpn_loss": avg_rpn_loss,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                
                local_logger.log(
                    log_type="iteration",
                    iteration=global_iteration,
                    iter_total_loss=avg_total_loss,
                    iter_cls_loss=avg_cls_loss,
                    iter_box_loss=avg_box_loss,
                    iter_object_loss=avg_object_loss,
                    iter_rpn_loss=avg_rpn_loss,
                    learning_rate=scheduler.get_last_lr()[0]
    )
                
                accelerator.print(f"Iteration {global_iteration}: Loss={avg_total_loss:.4f} (Cls:{avg_cls_loss:.4f}, Box:{avg_box_loss:.4f}, Object:{avg_object_loss:.4f}, RPN:{avg_rpn_loss:.4f})")
                
                # Reset running averages
                running_total_loss = 0.0
                running_cls_loss = 0.0
                running_box_loss = 0.0
                running_object_loss = 0.0
                running_rpn_loss = 0.0
                running_count = 0

            # Evaluate mAP every 200 iterations
            #if global_iteration % 15 == 0 and accelerator.is_main_process: 
            if global_iteration % 10000000 == 0:
                accelerator.print(f"Evaluating mAP at iteration {global_iteration}...")
                model.eval()
                with torch.no_grad():
                    mAP_score = evaluate_MAP_full(model, test_dataloader, device=model_device,accelerator=accelerator)
                
                if args.use_wandb:
                    wandb.log({
                        "iteration": global_iteration,
                        "mAP": mAP_score
                    })
                
                mAP_score = {}
                for k, v in mAP_score.items():
                    if v.numel() == 1:
                        mAP_score[k] = float(v.item())
                    else:
                        # Convert to float if needed, then take mean
                        mAP_score[k] = float(v.float().mean())

                local_logger.log(
                    log_type="mAP_evaluation",
                    iteration=global_iteration,
                    mAP=mAP_score
                )
                
                accelerator.print(f"Iteration {global_iteration}: mAP = {mAP_score}")
                accelerator.wait_for_everyone()
                model.train() 
        ## Only when GPUs are being synchronised (When all grad accumulation is done) store metrics ##
    epoch_avg_loss = np.mean(batch_loss_list) if batch_loss_list else 0.0
    epoch_avg_cls_loss = np.mean(batch_loss_cls_list) if batch_loss_cls_list else 0.0
    epoch_avg_box_loss = np.mean(batch_loss_box_reg_list) if batch_loss_box_reg_list else 0.0
    epoch_avg_object_loss = np.mean(batch_loss_objectness_list) if batch_loss_objectness_list else 0.0
    epoch_avg_rpn_loss = np.mean(batch_loss_rpn_list) if batch_loss_rpn_list else 0.0

    # Evaluate mAP at end of each epoch
    # if accelerator.is_main_process:
    accelerator.print(f"Evaluating mAP at end of epoch {epoch+1}...")
    model.eval()
    with torch.no_grad():
        epoch_mAP = evaluate_MAP_full(model, test_dataloader, device=model_device,accelerator=accelerator)
    
    # Plot predictions after each epoch
    if accelerator.is_main_process:
        vis_save_dir = os.path.join(path_to_experiment, "visualizations")
        try:
            plot_predictions_after_epoch(
                model=model,
                dataset=test_dataset,  # Use test dataset
                device=model_device,
                epoch=epoch,
                sample_idx=120,  # Fixed sample to track progress
                save_dir=vis_save_dir,
                score_threshold=0.1
            )
            accelerator.print(f"✓ Prediction visualization saved for epoch {epoch+1}")
        except Exception as e:
            accelerator.print(f"⚠️  Failed to save visualization: {e}")

    
    accelerator.print(f"mAP: {epoch_mAP}")
    # Log epoch results
    if args.use_wandb:
        wandb.log({
            "epoch": epoch,
            "train/total_loss": epoch_avg_loss,
            "train/cls_loss": epoch_avg_cls_loss,
            "train/box_loss": epoch_avg_box_loss,
            "train/object_loss": epoch_avg_object_loss,
            "train/rpn_loss": epoch_avg_rpn_loss,
            "epoch/mAP": epoch_mAP,
            "learning_rate": scheduler.get_last_lr()[0]
        })
    for k, v in epoch_mAP.items():
        if v.numel() == 1:
            epoch_mAP[k] = float(v.item())
        else:
            # Convert to float if needed, then take mean
            epoch_mAP[k] = float(v.float().mean())
    # Log to local logger
    local_logger.log(
        log_type="epoch",
        epoch=epoch,
        train_total_loss=epoch_avg_loss,
        train_cls_loss=epoch_avg_cls_loss,
        train_box_loss=epoch_avg_box_loss,
        train_object_loss=epoch_avg_object_loss,
        train_rpn_loss=epoch_avg_rpn_loss,
        mAP=epoch_mAP,
        learning_rate=scheduler.get_last_lr()[0]
    )

    # Print epoch summary
    accelerator.print(f"\nEpoch {epoch+1}/{args.epochs}")
    accelerator.print(f"Train Loss: {epoch_avg_loss:.4f}")
    accelerator.print(f"  Cls: {epoch_avg_cls_loss:.4f}")
    accelerator.print(f"  Box: {epoch_avg_box_loss:.4f}")
    accelerator.print(f"  Object: {epoch_avg_object_loss:.4f}")
    accelerator.print(f"  RPN: {epoch_avg_rpn_loss:.4f}")
    accelerator.print(f"mAP: {epoch_mAP}")
    accelerator.print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    accelerator.print("-" * 80)

    ### Checkpoint Model ###
    path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
    
    # Save checkpoint with epoch, scheduler state, LR, and model
    if accelerator.is_main_process:
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'current_lr': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else None,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        os.makedirs(path_to_checkpoint, exist_ok=True)
        checkpoint_file = os.path.join(path_to_checkpoint, 'checkpoint_state.pth')
        torch.save(checkpoint_dict, checkpoint_file)
        accelerator.print(f"✓ Checkpoint saved at epoch {epoch+1} to {checkpoint_file}")
    else:
        accelerator.print(f"Checkpoint saved at epoch {epoch+1}")

accelerator.end_training()
