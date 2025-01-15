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
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image

# Load image and model
img_path = r"C:\Users\rafam\Desktop\Trabajo\Investigation\yolov11-explainable-ai\data\test\images\1478020251695726207_jpg.rf.39e2de33289adecec9b823c96dbe53a3.jpg"
img_original = cv2.imread(img_path)  # Using OpenCV for image loading
img_resized = cv2.resize(img_original, (640, 640))
img_float32 = np.float32(img_resized) / 255
transform = transforms.ToTensor()
tensor = transform(img_float32).unsqueeze(0)

# Load YOLO model
model = YOLO(r"C:\Users\rafam\Desktop\Trabajo\Investigation\yolov11-explainable-ai\models\yolov11_models_m\weights\best.pt")
model.eval()
model.cpu()  # Moves the model to CPU
target_layers = [model.model.model[-2]]

# Run inference and draw bounding boxes
results = model(img_path)  # Perform inference on the image path

# Display and extract the image with bounding boxes
for result in results:
    # Draw bounding boxes directly on the image
    img_with_boxes = result.plot()  # Returns image with bounding boxes drawn

# Apply dff
img_with_boxes = cv2.resize(img_with_boxes, (640, 640))
img_with_boxes = np.float32(img_with_boxes) / 255
dff = DeepFeatureFactorization(model=model, target_layer=model.model.model[-2])
concepts, batch_explanations = dff(tensor, 8)
visualization = show_factorization_on_image(img_with_boxes, batch_explanations[0], image_weight=0.7)



# Display the final image
plt.imshow(visualization)
plt.axis('off')  # Hide axes for a clean display
plt.show()
