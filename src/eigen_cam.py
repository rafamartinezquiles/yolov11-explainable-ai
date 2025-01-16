from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import matplotlib.pyplot as plt

# Load image and model
# img_path = "full_path_to_the_image"
img_original = cv2.imread(img_path)  # Using OpenCV for image loading
img_resized = cv2.resize(img_original, (640, 640))
img_float32 = np.float32(img_resized) / 255
transform = transforms.ToTensor()
tensor = transform(img_float32).unsqueeze(0)

# Load YOLO model
# model = YOLO("full_path_to_the_model_weight")
model.eval()
model.cpu()  # Moves the model to CPU
target_layers = [model.model.model[-2]]

# Run inference and draw bounding boxes
results = model(img_path)  # Perform inference on the image path

# Display and extract the image with bounding boxes
for result in results:
    # Draw bounding boxes directly on the image
    img_with_boxes = result.plot()  # Returns image with bounding boxes drawn

# Ensure img_with_boxes is in the correct color space for display
orig_plot = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

# Apply Grad-CAM
cam = EigenCAM(model, target_layers)
grayscale_cam = cam(tensor, targets=[])[0, :, :]

# Resize Grad-CAM result to match image dimensions
grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_plot.shape[1], orig_plot.shape[0]))

# Superimpose Grad-CAM heatmap on the image
cam_image = show_cam_on_image(np.float32(orig_plot) / 255, grayscale_cam_resized, image_weight=0.5)

# Display the final image
plt.imshow(cam_image)
plt.axis('off')  # Hide axes for a clean display
plt.show()