# Analyzing Pixel Contributions with Explainable AI on Autonomous Driving


## Overview and Background


## Table of Contents
```
BibObjectDetection
|__ images
|   |__ results_accuracy.png 
|   |__ threshold.jpg
|   |__ yolo_application.png
|__ weights
|   |__ BDBD
|   |   |__ yolov8l.pt
|   |   |__ yolov8m.pt 
|   |   |__ yolov8s.pt 
|   |   |__ yolov8n.pt 
|   |__ People
|   |   |__ yolov8l.pt
|   |   |__ yolov8m.pt 
|   |   |__ yolov8s.pt 
|   |   |__ yolov8n.pt 
|   |__ SVHN
|   |   |__ yolov8l.pt
|   |   |__ yolov8m.pt 
|   |   |__ yolov8s.pt 
|   |   |__ yolov8n.pt 
|__ labels
|   |__ labels_test
|   |   |__ all the labels in txt format
|   |__ labels_train
|   |   |__ all the labels in txt format
|__ src
    |__ create_csv.py
    |__ create_yaml.py
    |__ data_augmentation.py
    |__ image_prediction.py
    |__ move_png_files.py
    |__ train.py
    |__ video_prediction.py
README.md
requirements.txt
```

## Getting started

### Resources used


### Installing
The project is deployed in a local machine, so you need to install the next software and dependencies to start working:

1. Create and activate the new virtual environment for the project

```bash
conda create --name yolov11-explainable-ai python=3.10
conda activate yolov11-explainable-ai
```

2. Clone repository

```bash
git clone https://github.com/rafamartinezquiles/yolov11-explainable-ai.git
```

3. In the same folder that the requirements are, install the necessary requirements

```bash
cd yolov11-explainable-ai
pip install -r requirements.txt
```

### Setup
It is worth noting that the "Udacity Self Driving Car Dataset" provides functionality to download it in YOLOv11 format, which is recommended.

1. Retrieve the "Udacity Self Driving Car Dataset" in YOLOv11 format from the provided [link](https://public.roboflow.com/object-detection/self-driving-car/2). Download it as a zip file and ensure to place it within the main folder of the cloned repository named yolov11-explainable-ai.

```bash
mv /path/to/source /path/to/destination
```

2. Inside the cloned repository, execute the following command in order to unzip the "Udacity Self Driving Car Dataset" necessary for the project elaboration.

```bash
tar -xf "Self Driving Car.v2-fixed-large.yolov11.zip" -C Udacity
```

3. Since the data is not divided into training and test, we will run the following python file that will divide the data into 70% for training and the remaining 30% for test taking into account the images and labels.

```bash
python src/split_train_test.py
```