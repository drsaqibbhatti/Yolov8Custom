import torch
import random
from datetime import date
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.YOLO_V8 import yolov8_N
from utils.util import ComputeLoss, ComputeIoU, EMA, clip_gradients, setup_seed, save_checkpoint, load_checkpoint
from utils.ObjDataLoader import ObjDataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



# Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
setup_seed()

# Data transformation (no resizing, handled in dataset class)
transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to PyTorch tensor
])




def train(load_previous_model=True):
    # Device configuration
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"The device is: {device}")

    # Set random seeds for reproducibility
    random.seed(777)
    torch.manual_seed(777)
    if USE_CUDA:
        torch.cuda.manual_seed_all(777)
    imageHeight = 512
    imageWidth = 512
    # Hyperparameters
    batchSize = 32
    learningRate = 0.001
    epochs = 50
    target_iou = 0.999

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load datasets
    trainDataset = ObjDataLoader(
                                root="/mnt/d/hvs/Hyvsion_Projects/Pre_training/Pre_Training_Personal/Aerial Maritime.v9-tiled.coco/train", 
                                annFile="/mnt/d/hvs/Hyvsion_Projects/Pre_training/Pre_Training_Personal/Aerial Maritime.v9-tiled.coco/train/_annotations.coco.json",
                                transform=transform,
                                img_size=(imageHeight, imageWidth),
                                normalize_boxes=True)

    
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False, drop_last=True, collate_fn=lambda x: tuple(zip(*x)))



    # Initialize model
    num_classes = 5
    model = yolov8_N(num_classes=num_classes)
    

    # # Multi-GPU support
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = DataParallel(model)

    model = model.to(device)

    # Loss and optimizer
    loss_fn = ComputeLoss(model, params={'cls': 0.5, 'box': 1.0, 'dfl': 0.5})
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    ema = EMA(model)
    amp_scaler = torch.cuda.amp.GradScaler() 
    # Track best IoU
    best_iou = 0.0
    best_model_path = None

    # Directory to save model
    Base_dir = '/home/saqib/Simulations_Ubuntu/Pre_training_personal/runs'
    model_name = "yolov8_N"
    build_date = str(date.today())
    model_dir = os.path.join(Base_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    existing_runs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    run_number = len(existing_runs) + 1
    run_dir = os.path.join(model_dir, f"run_{run_number}")
    os.makedirs(run_dir)
    metrics = []

    # Check if a checkpoint exists
    checkpoint_dir = os.path.join(run_dir, 'Checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = 0
    if load_previous_model:
        previous_run_number = run_number - 1
        previous_run_dir = os.path.join(model_dir, f"run_{previous_run_number}")
        previous_checkpoint_dir = os.path.join(previous_run_dir, 'Checkpoints')

        if os.path.exists(previous_checkpoint_dir):
            checkpoint_files = sorted([f for f in os.listdir(previous_checkpoint_dir) if f.startswith('checkpoint_epoch_')])
            if checkpoint_files:
                last_checkpoint = checkpoint_files[-1]
                checkpoint_path = os.path.join(previous_checkpoint_dir, last_checkpoint)
                start_epoch = load_checkpoint(checkpoint_path, model, optimizer) + 1
                print(f"Loaded checkpoint from {previous_checkpoint_dir} at epoch {start_epoch}")
            else:
                print("No checkpoint files found in the previous run.")
        else:
            print("No previous run found to load a checkpoint from.")
    else:
        print("Starting from scratch, not loading any previous model.")
    saving_metrics = []
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_ious = 0.0
        optimizer.zero_grad()

        with tqdm(total=len(trainLoader), desc=f"Epoch {epoch}/{epochs-1}", unit="batch") as pbar:
            for images, targets in trainLoader:
                images = torch.stack([img.to(device) for img in images], dim=0)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass with AMP
                with torch.cuda.amp.autocast():
                    outputs = model(images)

                    loss, pred_boxes, target_boxes = loss_fn(outputs, targets)  # Calculate the loss

                # Backward pass with scaled gradients
                amp_scaler.scale(loss).backward()

                # Clip gradients and step optimizer
                amp_scaler.unscale_(optimizer)  # Unscale gradients before clipping
                clip_gradients(model, max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()  # Clear gradients

                ema.update(model)  # Update EMA

                train_loss += loss.item()
                
                with torch.no_grad():
                    ious = ComputeIoU(pred_boxes, target_boxes)
                    batch_mean_iou = ious.mean().item()
                    train_ious+= batch_mean_iou

                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), ious=batch_mean_iou)
        train_loss /= len(trainLoader)

        train_mean_iou =train_ious/ len(trainLoader)

                    
        print(f"Train Loss:{train_loss:.9f}, Train IoU: {train_mean_iou:.9f}")

        # Save checkpoint after each epoch
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_checkpoint(checkpoint_state, model, checkpoint_dir, epoch)

        # Save best model based on IoU
        if train_mean_iou > best_iou:
            if best_model_path is not None:
                os.remove(best_model_path)
            best_iou = train_mean_iou
            best_model_path = os.path.join(run_dir, f"{build_date}_best_epoch_{epoch}_iou_{best_iou:.4f}.pth")
            torch.save(ema.ema.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch} with train IoU : {best_iou:.4f}")

        # Early stopping based on IoU
        if train_mean_iou >= target_iou:
            print(f"Target IoU reached at epoch {epoch}. Stopping training.")
            break

        # Save metrics every 10 epochs

        saving_metrics.append({
            'Epoch': epoch,
            'TrainLoss': round(train_loss,4),
            'IoU': round(train_mean_iou,9)
        })
        # Save metrics every 10 epochs or at the last epoch
        if epoch % 10 == 0 or epoch == epochs - 1:
            metrics_df = pd.DataFrame(saving_metrics)
            csv_path = os.path.join(run_dir, f"training_metrics_epoch_{epoch}.csv")
            metrics_df.to_csv(csv_path, index=False)
            plot_metrics(metrics_df, run_dir, epoch)

    # Save final model
    final_model_path = os.path.join(run_dir, f"{build_date}_last_epoch_{epoch}_train_iou_{train_mean_iou:.4f}.pth")
    torch.save(ema.ema.state_dict(), final_model_path)
    print(f"Saved final model after epoch {epoch}")

def plot_metrics(metrics_df, run_dir, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['IoU'], label='IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title(f'IoU (Up to Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, f'IoU_epoch_{epoch}.png'))
    plt.close()

if __name__ == '__main__':
    train()
