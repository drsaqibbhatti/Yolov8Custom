import torch
import os
from utils.util import box_iou, compute_ap, non_max_suppression
from utils.ObjDataLoader import ObjDataLoader
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model.YOLO_V8 import yolov8_N
from PIL import ImageDraw
import numpy as np
from tqdm import tqdm
import os
from PIL import ImageDraw, Image

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "/mnt/d/hvs/Hyvsion_Projects/Pre_training/Pre_Training_Personal/Aerial Maritime.v9-tiled.coco/valid_evaluated_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Load the validation dataset
imageHeight = 512
imageWidth = 512
shape = (imageHeight, imageWidth)
transform = transforms.Compose([
    transforms.ToTensor(),
])

validDataset = ObjDataLoader(
    root="/mnt/d/hvs/Hyvsion_Projects/Pre_training/Pre_Training_Personal/Aerial Maritime.v9-tiled.coco/valid",
    annFile="/mnt/d/hvs/Hyvsion_Projects/Pre_training/Pre_Training_Personal/Aerial Maritime.v9-tiled.coco/valid/_annotations.coco.json",
    transform=transform,
    img_size=(imageHeight, imageWidth),
    normalize_boxes=True
)

validLoader = DataLoader(validDataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=lambda x: tuple(zip(*x)))

num_classes = 5
class_names=['boat', 'car','dock','jetski','lift']
# Load the model
model_path = "/home/saqib/Simulations_Ubuntu/Pre_training_personal/runs/yolov8_N/run_1/2024-11-19_last_epoch_49_train_iou_0.7988.pth"

model = yolov8_N(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device) 
model.eval()

# Configure metrics tracking
iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # IoU thresholds for mAP@0.5:0.95
n_iou = iou_v.numel()

m_pre = 0.
m_rec = 0.
map50 = 0.
mean_ap = 0.
metrics = []

total_iou = 0.0
total_boxes = 0

with torch.no_grad():
    for i, (images_val, targets_val) in enumerate(tqdm(validLoader, total=len(validLoader), desc="Validating", unit="batch")):
        # Prepare images and targets
        images_val = torch.stack([img.to(device) for img in images_val], dim=0)
        targets_val = [{k: v.to(device).float() for k, v in t.items()} for t in targets_val]

        _, _, height, width = images_val.shape  # Get the actual image dimensions after resizing

        # Get model predictions
        outputs = model(images_val)

        # Apply Non-Maximum Suppression (NMS)
        outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.65)


        # Convert ground truth boxes from normalized [x_center, y_center, width, height] to pixel [x_min, y_min, x_max, y_max]
        for target in targets_val:
            boxes = target['boxes']
            x_center, y_center, box_width, box_height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

            # Convert to pixel values
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Convert to [x_min, y_min, x_max, y_max]
            x_min = x_center - (box_width / 2)
            y_min = y_center - (box_height / 2)
            x_max = x_center + (box_width / 2)
            y_max = y_center + (box_height / 2)

            # Update target boxes with the converted values
            target['boxes'] = torch.stack([x_min, y_min, x_max, y_max], dim=1)

            
        # Convert to CPU for drawing
        outputs = [output.cpu() for output in outputs]
        targets_val = [{k: v.cpu() for k, v in t.items()} for t in targets_val]
        
        for j, output in enumerate(outputs):
            if output is None:
                continue

            labels = targets_val[j]['boxes']
            class_labels = targets_val[j]['labels'].long()
            
            # Predictions are already in [x_min, y_min, x_max, y_max] format
            detections = output.clone()
            predicted_classes = detections[:, 5].long()
            predicted_boxes = detections[:, :4]
            
            # Convert image tensor to PIL for drawing
            img = images_val[j].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
            img = img[:, :, [2, 1, 0]]  # Swap channels from BGR to RGB
            img = (img * 255).astype(np.uint8)  # Convert to uint8
            pil_img = Image.fromarray(img)
            
            draw = ImageDraw.Draw(pil_img)

            # Evaluate metrics
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)
            if output.shape[0] == 0:
                if labels.shape[0] > 0:
                    metrics.append((correct, *torch.zeros((3, 0)).to(device)))
                continue

            # Compute IoU between ground truth and detections
            iou = box_iou(labels, predicted_boxes)
            # print(f"IoU Matrix for Image {i}:\n{iou.cpu().numpy()}")

            
            if iou.numel() == 0:
                continue

            # Calculate the overall IoU for the current image by finding the maximum IoU for each ground truth box
            max_ious_per_gt = torch.max(iou, dim=1)[0]  # Find max IoU for each ground truth box
            image_iou_sum = max_ious_per_gt.sum().item()  # Sum of IoUs for this image
            num_gt_boxes = len(max_ious_per_gt)  # Number of ground truth boxes

            # Update totals for overall IoU calculation
            total_iou += image_iou_sum
            total_boxes += num_gt_boxes
            
            

            for box in labels:
                x_min, y_min, x_max, y_max = box.tolist()
                draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

            # Draw predicted boxes (red)
            for k, pred_box in enumerate(predicted_boxes):
                x_min, y_min, x_max, y_max = pred_box.tolist()
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)    
                if iou.numel() > 0 and k < iou.shape[1]:
                    max_iou_value = iou[:, k].max().item()
                    iou_text = f"IoU: {max_iou_value:.2f}"
                else:
                    iou_text = "IoU: N/A"
        
                class_name = class_names[predicted_classes[k].item()]
                score_text = f"{class_name} : ,Conf: {detections[k, 4]:.4f}, {iou_text}"
                draw.text((x_min, y_min - 10), score_text, fill="red")
            
            image_filename = os.path.join(output_dir, f"overlay_image_{i}_{j}.png")
            pil_img.save(image_filename)
            
            
            for iou_idx in range(len(iou_v)):
                iou_threshold = iou_v[iou_idx]
                matches = torch.zeros_like(correct[:, iou_idx])  # Store matches for each detection

                for k, detection in enumerate(detections):
                    # Get IoU values for the current detection against all ground truth boxes
                    iou_values = iou[:, k]

                    # Find the ground truth box with the highest IoU above the threshold
                    max_iou, max_iou_index = torch.max(iou_values, dim=0)

                    if max_iou >= iou_threshold:
                        # Check if class labels match as well
                        if class_labels[max_iou_index] == predicted_classes[k]:

                            matches[k] = True

                correct[:, iou_idx] = matches
            metrics.append((correct.cpu(), detections[:, 4].cpu(), predicted_classes.cpu(), class_labels.cpu()))


# Compute metrics
metrics = [torch.cat([t.to('cpu') for t in x], 0).numpy() for x in zip(*metrics)]  # Convert metrics to numpy


if len(metrics) and metrics[0].any():
    tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

print(f"Saved overlay image: {output_dir}")
# Print results
print(f'Precision: {m_pre:.9f}, Recal: {m_rec:.9f},mAP@0.5: {map50:.9f} mean_ap: {mean_ap:.9f}')

overall_iou = total_iou / total_boxes if total_boxes > 0 else 0
print(f"Overall IoU: {overall_iou}")
