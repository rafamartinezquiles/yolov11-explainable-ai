# Analyzing Pixel Contributions with Explainable AI on Autonomous Driving
Using machine learning algorithms, including Ultralytics, Torch, OpenCV and NIVIDIA's CUDA to see what pixels affect the most in the context of autonomous driving during object detection.

## Overview and Background
Detecting and interpreting critical image features for autonomous driving tasks is a complex and pivotal challenge. This project focuses on explainable AI techniques applied to image classification and object detection, offering insights into which pixel-level features most significantly contribute to model predictions. By leveraging state-of-the-art neural networks, including YOLOv11, and advanced visualization methods, this repository demonstrates how explainability can enhance trust and performance assessment in autonomous driving models.

![](images/overview_and_background.png)

## Table of Contents
```
yolov11-explainable-ai
|__ images
    |__ yolov11_txt_format.jpeg
    |__ model_performance_comparison.png
    |__ timing_analysis_heatmap.png
    |__ dff.png
    |__ eigen_cam.png
    |__ combination_dff_original.png
    |__ overview_and_background.png
|__ src
    |__ split_train_test.py
    |__ prediction.py
    |__ computational_efficiency_matrix.py
    |__ graph_evaluation_metrics.py
    |__ evaluation_metrics.py
    |__ dff.py
    |__ eigen_cam.py
    |__ combination_dff_original.py
    |__ pytorch-grad-cam
        |__ examples
        |__ pytorch_grad_cam
        |__ tests
        |__ tutorials
        |__ usage_examples
        |__ cam.py
        |__ setup.py
README.md
requirements.txt
```

## Getting started

### Resources used
A high-performance desktop computer equipped with an AMD Ryzen Threadripper PRO 5975WX 32-Core CPU and a powerful NVIDIA GeForce RTX 4090 GPU (24GB RAM) has been utilized for the training. This system is further supported by 128GB of RAM, ensuring robust performance for demanding computational tasks. The GPU's advanced capabilities, including parallel processing, have been leveraged to optimize training time.

On the other hand, an Acer Nitro AN515-55 laptop powered by an Intel® Core™ i7-10750H CPU @ 2.60GHz and equipped with 48GB of RAM has been utilized for the rest of operations. This high-performance system ensures efficient handling of the prediction phase and evaluation metric computations for various tasks. The laptop's 64-bit operating system and x64-based processor architecture support modern computational operations, including processing predictions, generating evaluation metrics, and managing large datasets.

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

In case of not having the necessary time or resources to train the neural networks, you can access the weights and training information of the neural networks in the following [link](https://drive.google.com/drive/folders/1M1R6dTDRGQSBAA3VU1CibKp3VIcQFow9?usp=drive_link).

## Extracting Labels After Prediction
This section covers the process of executing a script to create a new folder containing labels associated with the predictions from the test folder. Each label file will share the same name as its corresponding original image but will use a .txt extension instead of the image extension (e.g., .jpg or .png). This format allows for easy comparison with the original labels to compute evaluation metrics. The script can be executed as follows:

```bash
python src/prediction.py --model complete_path\yolov11_models_s\weights\best.pt --source complete_path\test\images
```

Executing this command yields several key results. First, it provides the average times in milliseconds for preprocessing, inference, and postprocessing, offering insights into the model's efficiency. Additionally, it outputs the total number of tags detected during the process. Finally, it generates a folder containing all the detected tags in .txt format, which can be used subsequently to calculate the desired evaluation metrics.

## Evaluation Metrics
For evaluating the performance of our model trained on this dataset, the following evaluation metrics are appropriate and their meaning will be the following. But before going into each of the evaluation metrics, it is necessary to highlight the way in which the labels are formed in order to understand how they work.

The labels in this dataset adhere to the YOLOv11 TXT format, which ensures consistency and compatibility with the YOLO framework. Each image is associated with a .txt file containing a line for each bounding box. The structure of each row is:

```bash
class_id center_x center_y width height
```

![](images/yolov11_txt_format.jpeg)


Also, the values are normalized ensuring that the bounding box coordinates are independent of image resolution, scaling them to fall within the range of 0 to 1.

### Mean Average Precision (mAP)
Mean Average Precision (mAP) is a widely used metric in object detection that evaluates the overall performance of a model in identifying objects. It combines precision and recall by calculating the average precision across all object classes. A high mAP value indicates that the model consistently detects objects correctly across multiple categories, with both high precision (few false positives) and high recall (few false negatives). Conversely, a low mAP suggests that the model struggles with either identifying objects or avoiding false positives, making it an important indicator of model effectiveness in real-world applications.

### Precision and Recall 
Precision and recall are fundamental metrics that assess a model's ability to detect relevant objects. Precision measures the proportion of true positives (correct detections) out of all predicted positives, representing how accurate the model is when it claims to have detected an object. Recall, on the other hand, measures the proportion of true positives out of all actual objects, indicating how well the model captures all the relevant instances. In object detection for autonomous driving, precision and recall are crucial for ensuring that the model identifies objects accurately while minimizing missed detections and false positives.

### F1 Score
The F1 score is a balanced metric that combines precision and recall into a single value, offering a more comprehensive view of a model's performance. It is the harmonic mean of precision and recall, where a higher F1 score indicates a model that performs well in both detecting objects accurately and capturing as many relevant objects as possible. In autonomous driving applications, a high F1 score is essential because it ensures that the object detection system is both reliable and comprehensive, making it a critical measure for evaluating model performance.


### Calculation with the code
This code is designed to evaluate the performance of object detection models by comparing predicted bounding boxes with ground truth bounding boxes. It parses YOLO format label files, computes Intersection over Union (IoU) to match predicted boxes to ground truth boxes, and then calculates precision, recall, F1 score, and mean Average Precision (mAP). 

```bash
python src/evaluation_metrics.py <original_labels_folder> <predicted_labels_folder>
```

## Eigen-Cam: Visualizing Feature Importance Without Gradients
EigenCAM is a powerful visualization technique for understanding which parts of an image influence the decisions of a YOLO model. Unlike gradient-based methods, EigenCAM operates directly on the activations, making it gradient-free. This is particularly advantageous when working with the Ultralytics YOLO implementation, where obtaining gradients is not straightforward. Some of the key features are:

- **Blue Pixel Highlights:** EigenCAM visualizes important regions of the image by highlighting dominant spatial features. The blue pixels indicate areas with the greatest impact on the model's output, often correlating closely with the detected bounding boxes.
- **No Class Discrimination:** Unlike Grad-CAM, which associates regions with specific classes, EigenCAM simply identifies the most significant features. This method provides a broad understanding of feature importance but does not differentiate between categories.
- **Gradient-Free Activation Analysis:** EigenCAM uses the first principal component of the activation map to detect important spatial features. It analyzes the activation space without requiring gradients, simplifying the process and enhancing compatibility with YOLO.

Before executing the following command, follow these steps to modify the `src/eigen_cam.py` file:

1. Open `src/eigen_cam.py`.
2. Locate the lines:
   ```python
   # img_path = "full_path_to_the_image"
   # model = YOLO("full_path_to_the_model_weight")
3. Replace "full_path_to_the_image" and "full_path_to_the_model_weight" with the appropriate file paths.
4. Uncomment both lines by removing the # at the beginning:
5. Save the file.

Once these steps are completed, you are ready to execute the command below:

```bash
python src/eigen_cam.py 
```

![](images/eigen_cam.png)

This script will process the specified input image, apply EigenCAM, and display the resulting image with blue pixels highlighting the most significant spatial features contributing to the model’s predictions. The only limitation is that there is no Class-Specific Attribution.

## Deep Feature Factorization (DFF): Decomposing Feature Activations for Insights
Deep Feature Factorization (DFF) is a method that decomposes the activation space of a model into multiple spatial components, revealing diverse patterns and regions of importance in the image. Unlike methods that provide a single visualization, DFF extracts multiple activation components, making it ideal for understanding complex feature hierarchies in object detection models like YOLO. Some of the key features are:

- **Component-Based Visualization:** DFF factorizes the activations into multiple spatial components. In this implementation, the activations are divided into 8 distinct components, each highlighting different regions or features that influence the model’s predictions.
- **Gradient-Free Method:** Like EigenCAM, DFF does not require gradient computations, making it highly compatible with the Ultralytics YOLO framework, where obtaining gradients is non-trivial.
- **Multi-Faceted Feature Exploration:** Instead of focusing on a single most significant feature, DFF provides a broader, more detailed view of the activation space by identifying multiple activation patterns within a single image.

Before executing the following command, follow these steps to modify the `src/dff.py` file:

1. Open `src/dff.py`.
2. Locate the lines:
   ```python
   # img_path = "full_path_to_the_image"
   # model = YOLO("full_path_to_the_model_weight")
3. Replace "full_path_to_the_image" and "full_path_to_the_model_weight" with the appropriate file paths.
4. Uncomment both lines by removing the # at the beginning:
5. Save the file.

Once these steps are completed, you are ready to execute the command below:

```bash
python src/dff.py
```

![](images/dff.png)

DFF offers several benefits and trade-offs when compared to other visualization methods like EigenCAM. One key advantage is its multi-component analysis, which provides a comprehensive breakdown of feature activations. Unlike EigenCAM, which focuses on dominant spatial features, DFF reveals separate feature clusters, allowing for deeper exploration of how different regions of an image influence the YOLO model’s predictions. However, a notable trade-off is that DFF, similar to EigenCAM, does not provide class-level information. While it effectively identifies significant spatial activation structures, it does not indicate which specific object category each component corresponds to, limiting its use for category-specific interpretability. Despite this limitation, DFF’s ability to uncover diverse feature activations makes it a valuable tool for enhancing understanding of complex model behavior.

## Combining DFF Results with the Original Image
It is possible to overlay the DFF (Deep Feature Factorization) visualization onto the original image to gain a more intuitive understanding of how different feature components align with the objects detected. However, it is important to consider that the aspect ratio of the DFF result and the original image may differ. This discrepancy can lead to slight misalignments when combining the two images, making it appear as though some elements do not perfectly match. To minimize this effect, careful resizing or aspect ratio adjustments may be necessary to improve alignment between the DFF visualization and the original image while preserving spatial accuracy. Despite these challenges, combining DFF results with the input image remains a powerful way to interpret model predictions by highlighting how feature clusters correspond to visual regions of interest.

Before executing the following command, follow these steps to modify the `src/combination_dff_original.py` file:

1. Open `src/combination_dff_original.py`.
2. Locate the lines:
   ```python
   # img_path = "full_path_to_the_image"
   # model = YOLO("full_path_to_the_model_weight")
3. Replace "full_path_to_the_image" and "full_path_to_the_model_weight" with the appropriate file paths.
4. Uncomment both lines by removing the # at the beginning:
5. Save the file.

Once these steps are completed, you are ready to execute the command below:

```bash
python src/combination_dff_original.py
```

![](images/combination_dff_original.png)

## Additional Task - Evaluation Metrics Comparison Graph
In this section we will show how to execute the code associated with the following representation:

![](images/model_performance_comparison.png)

This graph provides a clear comparison of the evaluation metrics across all YOLOv11 models. A quick analysis reveals that the extreme model outperforms the others in most metrics, with the nano model following closely behind. Interestingly, this highlights that a larger model does not always guarantee better performance. To obtain it, we should only execute the following command and we are all set.

```bash
python src/graph_evaluation_metrics.py
```

## Additional Task - Computational Efficiency Graph
In this section we will show how to execute the code associated with the following representation:

![](images/timing_analysis_heatmap.png)

This graph presents a detailed comparison of the computational efficiency across all YOLOv11 models. A closer look reveals that the preprocessing and postprocessing times remain nearly constant, irrespective of the model size, indicating that these stages are not influenced by the model used. However, the inference time per image significantly increases as the model size grows, which is expected due to the increased complexity and computational demands of larger models. This logical trend highlights the trade-off between model size and computational efficiency. To replicate these results, simply execute the following command, and you're ready to go.

```bash
python src/computational_efficiency_matrix.py
