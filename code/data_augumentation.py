import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

def augment_images(original_images_folder, augmented_images_folder):
    # Create the augmented images folder if it doesn't exist
    os.makedirs(augmented_images_folder, exist_ok=True)

    # Define the ImageDataGenerator with desired augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Iterate over the subfolders (classes) in the original images folder
    for class_folder in os.listdir(original_images_folder):
        class_folder_path = os.path.join(original_images_folder, class_folder)

        # Create a corresponding subfolder in the augmented images folder
        augmented_class_folder = os.path.join(augmented_images_folder, class_folder)
        os.makedirs(augmented_class_folder, exist_ok=True)

        # Iterate over the images in the current class folder
        for filename in os.listdir(class_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Load the image
                image_path = os.path.join(class_folder_path, filename)
                img = Image.open(image_path)

                # Convert the image to numpy array
                x = img.convert('RGB')
                x = x.resize((224, 224))
                x = np.array(x)

                # Expand dimensions to match the input shape of the generator
                x = np.expand_dims(x, axis=0)

                # Generate augmented images
                augmented_images = datagen.flow(x, batch_size=1)

                # Save augmented images
                for i, augmented_image in enumerate(augmented_images):
                    augmented_image = augmented_image.astype('uint8')
                    augmented_image = Image.fromarray(augmented_image[0])
                    augmented_image.save(os.path.join(augmented_class_folder, f'augmented_{i}_{filename}'))

                    # Specify the number of augmented images to generate for each original image
                    if i == 8:
                        break
original_images_folder = '/home/priyanka/CI-CD/images_split/train_folder'
augmented_images_folder = '/home/priyanka/CI-CD/images_split/aug_train_folder'
augment_images(original_images_folder, augmented_images_folder)