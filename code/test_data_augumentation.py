'''
In this example, we define a test function test_augment_images using the Pytest framework. We use the @pytest.fixture decorator to define fixtures for the paths of the original and augmented images folders. The augment_images function (which contains your code) is called to perform the augmentation.

Inside the test function, we verify that the augmented images are generated correctly by checking the existence of the augmented class folder, the number of augmented images, and the filenames of the augmented images. We also open the augmented images and check if they have the correct dimensions.
'''

import os
from PIL import Image
import numpy as np
import pytest

from data_augumentation import augment_images

@pytest.fixture
def original_images_folder():
    return "/home/priyanka/CI-CD/images_split/train_folder"

@pytest.fixture
def augmented_images_folder():
    return "/home/priyanka/CI-CD/images_split/aug_train_folder"

def test_augment_images(original_images_folder, augmented_images_folder):
    # Call the function to augment the images
    augment_images(original_images_folder, augmented_images_folder)

    # Check if the augmented images are generated correctly
    for class_folder in os.listdir(original_images_folder):
        augmented_class_folder = os.path.join(augmented_images_folder, class_folder)

        # Verify that the augmented class folder exists
        assert os.path.exists(augmented_class_folder)

        # Get the list of files in the original and augmented class folders
        original_files = os.listdir(os.path.join(original_images_folder, class_folder))
        augmented_files = os.listdir(augmented_class_folder)

        # Verify that the number of augmented images matches the expected count
        assert len(augmented_files) == (9 * len(original_files))

        # Verify that the augmented images are saved with the correct filenames
        for augmented_file in augmented_files:
            assert augmented_file.startswith('augmented_')
            assert augmented_file.endswith('.jpg') or augmented_file.endswith('.png')
            assert '_'.join(augmented_file.split('_')[2:]) in original_files

            # Verify that the augmented images can be opened and have the correct dimensions
            augmented_image = Image.open(os.path.join(augmented_class_folder, augmented_file))
            assert augmented_image.size == (224, 224)

# Run the tests
pytest.main()
