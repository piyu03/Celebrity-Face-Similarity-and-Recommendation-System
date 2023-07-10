import pytest
from predicting import *

def test_predict_similar_images():
    # Define the paths to the test image and model
    image_path = '/home/priyanka/CI-CD/images_split/test_folder/salman_khan/salman_khan_42.jpg'
    model_path = '/home/priyanka/CI-CD/celeb_face.h5'

    # Load the model
    model = load_model(model_path)

    # Get the class labels and indices mapping
    class_labels = list(train_generator.class_indices.keys())
    class_indices = train_generator.class_indices

    # Perform prediction using the function
    class_label, confidence = predict_similar_images(image_path)

    # Perform assertions to validate the prediction
    assert class_label is not None
    assert confidence is not None

    # Add additional assertions based on your specific requirements

    # Clean up any resources if needed

# Run the unit test
pytest.main(['-v', 'test_predicting.py'])
