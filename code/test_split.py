import os
import pytest
import shutil
from split import data_split
from sklearn.model_selection import train_test_split

# Define the paths for the training, validation, and test folders
train_folder = '/home/priyanka/CI-CD/images_split/train_folder'
valid_folder = '/home/priyanka/CI-CD/images_split/valid_folder'
test_folder = '/home/priyanka/CI-CD/images_split/test_folder'

@pytest.fixture
def main_folder():
    return '/home/priyanka/CI-CD/Images/'

def test_data_split(main_folder):
    # Call the data split function
    data_split(main_folder, train_folder, valid_folder, test_folder)

    # Check if the train, valid, and test folders are created
    assert os.path.exists(train_folder)
    assert os.path.exists(valid_folder)
    assert os.path.exists(test_folder)

    # Check if the expected number of subfolders (classes) are created
    expected_classes = ['kamal_haasan',
 'huma_qureshi',
 'salman_khan',
 'kirti_sanon',
 'kartik_aaryan',
 'vicky_kaushal',
 'Shahid_Kapoor',
 'Sara_Ali_Khan',
 'Sonam_Kapoor']  # Replace with your expected classes
    assert sorted(os.listdir(train_folder)) == sorted(expected_classes)
    assert sorted(os.listdir(valid_folder)) == sorted(expected_classes)
    assert sorted(os.listdir(test_folder)) == sorted(expected_classes)

    # Check if the images are properly split between train, valid, and test sets
    for class_folder in expected_classes:
        train_class_folder = os.path.join(train_folder, class_folder)
        valid_class_folder = os.path.join(valid_folder, class_folder)
        test_class_folder = os.path.join(test_folder, class_folder)

        # Verify that the train, valid, and test subfolders exist
        assert os.path.exists(train_class_folder)
        assert os.path.exists(valid_class_folder)
        assert os.path.exists(test_class_folder)

        # Count the number of images in each subfolder
        train_images = os.listdir(train_class_folder)
        valid_images = os.listdir(valid_class_folder)
        test_images = os.listdir(test_class_folder)

        # Verify the number of images in each split
        assert len(train_images) > 0
        assert len(valid_images) > 0
        assert len(test_images) > 0

        # Verify the total number of images is preserved
        total_images = len(train_images) + len(valid_images) + len(test_images)
        original_images = os.listdir(os.path.join(main_folder, class_folder))
        assert total_images == len(original_images)

# Run the tests
pytest.main()
