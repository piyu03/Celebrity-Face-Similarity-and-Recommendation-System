import os
import shutil
from sklearn.model_selection import train_test_split

def data_split(main_folder,train_folder,valid_folder,test_folder):
  # Create the training, validation, and test folders if they don't exist
  os.makedirs(train_folder, exist_ok=True)
  os.makedirs(valid_folder, exist_ok=True)
  os.makedirs(test_folder, exist_ok=True)

  # Define the minimum number of images per class for training and validation
  min_images_per_class = 1

  # Define the test size for the final split (e.g., 0.2 for 20% of the data)
  test_size = 0.2

  # Get the list of subfolders (classes) in the main folder
  class_folders = os.listdir(main_folder)

  # Iterate over each class folder
  for class_folder in class_folders:
      # Get the path for the current class folder
      class_path = os.path.join(main_folder, class_folder)

      # Get the list of image files in the current class folder
      images = os.listdir(class_path)

      # Check if the number of images in the class is sufficient for splitting
      if len(images) < min_images_per_class:
          print(f"Not enough images in class {class_folder}. Skipping.")
          continue

      # Split the image files into training, validation, and test sets
      train_images, valid_test_images = train_test_split(images, test_size=test_size, random_state=42)
      #print("len of train_images  ",len(train_images))
      #print("len of valid_test_images   ",len(valid_test_images))
      valid_images, test_images = train_test_split(valid_test_images, test_size=test_size, random_state=42)
      #print("len of valid_images   ",len(valid_images))
      #print("len of test_images   ",len(test_images))

      # Create subfolders for the current class in the training, validation, and test folders
      train_class_folder = os.path.join(train_folder, class_folder)
      valid_class_folder = os.path.join(valid_folder, class_folder)
      test_class_folder = os.path.join(test_folder, class_folder)
      os.makedirs(train_class_folder, exist_ok=True)
      os.makedirs(valid_class_folder, exist_ok=True)
      os.makedirs(test_class_folder, exist_ok=True)

      # Move the training images to the respective class subfolder in the training folder
      for train_image in train_images:
          src_path = os.path.join(class_path, train_image)
          dst_path = os.path.join(train_class_folder, train_image)
          shutil.copyfile(src_path, dst_path)

      # Move the validation images to the respective class subfolder in the validation folder
      for valid_image in valid_images:
          src_path = os.path.join(class_path, valid_image)
          dst_path = os.path.join(valid_class_folder, valid_image)
          shutil.copyfile(src_path, dst_path)

      # Move the test images to the respective class subfolder in the test folder
      for test_image in test_images:
          src_path = os.path.join(class_path, test_image)
          dst_path = os.path.join(test_class_folder, test_image)
          shutil.copyfile(src_path, dst_path)

# Define the paths for the training, validation, and test folders
# train_folder = '/home/priyanka/CI-CD/images_split/train_folder'
# valid_folder = '/home/priyanka/CI-CD/images_split/valid_folder'
# test_folder = '/home/priyanka/CI-CD/images_split/test_folder'
# main_folder = '/home/priyanka/CI-CD/Images'

# data_split(main_folder,train_folder,valid_folder,test_folder)