import os
import shutil
import random

# Set up paths according to the new folder structure
base_path = 'Udacity/export'
images_path = os.path.join(base_path, 'images')
labels_path = os.path.join(base_path, 'labels')

train_images_path = 'Udacity/train/images'
val_images_path = 'Udacity/val/images'
test_images_path = 'Udacity/test/images'
train_labels_path = 'Udacity/train/labels'
val_labels_path = 'Udacity/val/labels'
test_labels_path = 'Udacity/test/labels'

# Create directories if they don't exist
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Get a list of all image filenames
image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
random.shuffle(image_files)

# Calculate split indices for 60/20/20 split
train_split_index = int(len(image_files) * 0.6)
val_split_index = int(len(image_files) * 0.8)

train_files = image_files[:train_split_index]
val_files = image_files[train_split_index:val_split_index]
test_files = image_files[val_split_index:]

# Function to move files
def move_files(file_list, source_dir, dest_dir, file_type='image'):
    for file_name in file_list:
        # Move image files
        shutil.move(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
        
        # Move corresponding label files if file_type is 'image'
        if file_type == 'image':
            label_name = file_name.replace('.jpg', '.txt').replace('.png', '.txt')  # Adjust for your image extensions
            label_path = os.path.join(labels_path, label_name)
            if os.path.exists(label_path):
                if dest_dir == train_images_path:
                    shutil.move(label_path, os.path.join(train_labels_path, label_name))
                elif dest_dir == val_images_path:
                    shutil.move(label_path, os.path.join(val_labels_path, label_name))
                elif dest_dir == test_images_path:
                    shutil.move(label_path, os.path.join(test_labels_path, label_name))

# Move training, validation, and testing images and their labels
move_files(train_files, images_path, train_images_path)
move_files(val_files, images_path, val_images_path)
move_files(test_files, images_path, test_images_path)

print("Files moved successfully!")


