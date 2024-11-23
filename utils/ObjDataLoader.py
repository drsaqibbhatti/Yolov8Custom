import os
import random
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import numpy as np
import csv 
class ObjDataLoader(Dataset):
    def __init__(self, root, annFile, transform=None, useHFlip=False, useVFlip=False, img_size=(512, 512), normalize_boxes=True):
        """
        Args:
            root (str): Path to the images directory.
            annFile (str): Path to the COCO annotations file.
            transform (callable, optional): Transformations to apply to images.
            useHFlip (bool): Enable horizontal flip augmentation.
            useVFlip (bool): Enable vertical flip augmentation.
            img_size (tuple): Target image size (height, width) for resizing.
        """
        self.coco = CocoDetection(root=root, annFile=annFile)
        self.transform = transform
        self.useHFlip = useHFlip
        self.useVFlip = useVFlip
        self.img_size = img_size  # Expected as (height, width)
        self.normalize_boxes = normalize_boxes

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, targets = self.coco[idx]
        orig_width, orig_height = img.size



            
        # Apply horizontal and vertical flips, if specified
        if self.useHFlip and random.random() > 0.5:
            img = ImageOps.mirror(img)
        if self.useVFlip and random.random() > 0.5:
            img = ImageOps.flip(img)

        # Resize image
        img = img.resize(self.img_size, Image.NEAREST)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)

        # Compute scaling factors for bounding boxes
        scale_x = self.img_size[1] / orig_width
        scale_y = self.img_size[0] / orig_height

        # Adjust bounding boxes based on scaling factors
        boxes = []
        labels = []

        for target in targets:
            x_min, y_min, width, height = target['bbox']
            x_max = x_min + width
            y_max = y_min + height

            # Scale bounding box coordinates to resized image dimensions
            x_min = x_min * scale_x
            y_min = y_min * scale_y
            x_max = x_max * scale_x
            y_max = y_max * scale_y

            # Convert to center format
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            if self.normalize_boxes:
                x_center /= self.img_size[1]  # Normalize by target width
                y_center /= self.img_size[0]  # Normalize by target height
                w /= self.img_size[1]         # Normalize width
                h /= self.img_size[0]         # Normalize height

            boxes.append([x_center, y_center, w, h])
            labels.append(target['category_id'] - 1)

            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

            
        # Convert to BGR if needed
        img = img[[2, 1, 0], :, :]  # Swap channels to BGR

        # Prepare target dictionary
        target = {"boxes": boxes, "labels": labels}

        return img, target


    def generate_csv(self, csv_filename):
        """
        Generates a CSV file containing image ID and category IDs for each image.
        
        Args:
            csv_filename (str): Name of the CSV file to save the data.
        
        Format:
            Each row contains [image_id, category_ids (comma-separated list)]
        """
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_id', 'category_ids'])  # Header

            for image_info in self.images:
                image_id = image_info['id']
                category_ids = set()  # Use a set to avoid duplicate category IDs
                
                if image_id in self.image_annotations:
                    for annotation in self.image_annotations[image_id]:
                        category_ids.add(annotation['category_id']-1)
                
                # Convert category IDs to a sorted list for consistency
                category_ids = sorted(category_ids)
                
                # Write image ID and category IDs as a comma-separated string
                writer.writerow([image_id, ",".join(map(str, category_ids))])
        
        print(f"CSV file saved as {csv_filename}")
