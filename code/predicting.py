import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

# Preprocess the data using data augmentation for training and rescaling for validation
train_dir = '/home/priyanka/CI-CD/images_split/aug_train_folder'

# Define the parameters
image_size = (224, 224)
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


# Load the training and validation data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

# Load the saved model for prediction
loaded_model = load_model('/home/priyanka/CI-CD/celeb_face.h5')

# Get the class labels and indices mapping
class_labels = list(train_generator.class_indices.keys())
class_indices = train_generator.class_indices

# Function to predict similar images
def predict_similar_images(image_path):
    img = load_img(image_path, target_size=image_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = loaded_model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]  # Use class_labels instead of class_indices
    confidence = prediction[0][class_index]

    return class_label, confidence

# Example usage:
image_path = '/home/priyanka/CI-CD/images_split/test_folder/salman_khan/salman_khan_42.jpg'
class_label, confidence = predict_similar_images(image_path)
print(f"The image is similar to the celebrity: {class_label} (confidence: {confidence})")