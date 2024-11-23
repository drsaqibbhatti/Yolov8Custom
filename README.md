# Yolov8

conda create -n YOLO python=3.8

conda activate YOLO

conda install pytorch torchvision torchaudio cudatoolkit=10.2

pip install opencv-python==4.5.5.64

pip install tqdm

# Dataset Documentation

## Dataset Structure

The dataset should follow this directory structure:


Download the dataset with coco json format
The boxes are normalized in corner format and converted to box format using dataloader
├── dataset

      ├── train2017
      
            ├── coco_json_labels.json
      
            ├── 1111.jpg
            
            ├── 2222.jpg
      
      ├── val2017
      
            ├── coco_json_labels.json
            
            ├── 1111.jpg
            
            ├── 2222.jpg


# Reference
https://github.com/ultralytics/yolov5

https://github.com/ultralytics/ultralytics

https://docs.ultralytics.com/models/yolov8

https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md

https://public.roboflow.com/object-detection
