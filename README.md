# Yolov8Custom
conda create -n YOLO python=3.8
conda activate YOLO
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install tqdm


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
