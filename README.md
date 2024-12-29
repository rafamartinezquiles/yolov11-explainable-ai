# Analyzing Pixel Contributions with Explainable AI on Autonomous Driving


## Overview and Background


## Table of Contents
```
yolov11-explainable-ai
|__ images
|__ src
    |__ split_train_test.py
README.md
```

## Getting started

### Resources used
A high-performance desktop computer equipped with an AMD Ryzen Threadripper PRO 5975WX 32-Core CPU and a powerful NVIDIA GeForce RTX 4090 GPU (24GB RAM) has been utilized. This system is further supported by 128GB of RAM, ensuring robust performance for demanding computational tasks. The GPU's advanced capabilities, including parallel processing, have been leveraged to optimize training time.

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
mkdir Udacity
tar -xf "Self Driving Car.v2-fixed-large.yolov11.zip" -C Udacity
```

3. Since the data is not divided into training, test and validation, we will run the following python file that will divide the data into 60% for training, 20% for validation and the remaining 20% for test taking into account the images and labels.

```bash
python src/split_train_test.py
```

4. A particularity that YOLOv11 has, being quite new, is the fact that in the .yaml file in which we specify the path to the training, validation and test data, we have to specify the complete path. For it, in this step, we open the file “data.yaml” and we modify the alternative routes so that they are complete. An example is the following:
```bash
train: c:\Users\user_name\Desktop\yolov11-explainable-ai\Udacity\train\images
```

## Training of neural networks
The training of the neural networks will be accomplished by executing the following command, passing a series of arguments that define the characteristics of the neural network. The arguments to be specified are:

- **data:** This parameter represents the path leading to the .yaml file associated to the dataset. It should be the complete path
- **epochs:** Denotes the number of training epochs. 
- **imgsz:** Refers to the image size utilized during training.
- **batch:** Specifies the batch size utilized during training.
- **name:** Represents the name assigned to the neural network.
- **patience:** Number of epochs to wait without improvement in validation metrics before early stopping the training.

An example of the command used for training, incorporating these parameters, is as follows:

```bash
!yolo task=detect mode=train model=yolo11n.pt data=complete_path/yolov11-explainable-ai/SDC/data.yaml epochs=400 imgsz=640 batch=16 name=yolov11_models_n patience=10
```

In case of not having the necessary time or resources to train the neural networks, you can access the weights and training information of the neural networks in the following [Link](https://drive.google.com/drive/folders/1M1R6dTDRGQSBAA3VU1CibKp3VIcQFow9?usp=drive_link).