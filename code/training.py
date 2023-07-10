import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model
# Define the paths to the training and validation data
base_dir = '/home/priyanka/CI-CD/images_split'
train_dir = os.path.join(base_dir, 'aug_train_folder')
validation_dir = os.path.join(base_dir, 'valid_folder')

# Define the parameters
image_size = (224, 224)
batch_size = 16
num_epochs = 20

# Preprocess the data using data augmentation for training and rescaling for validation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and validation data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=image_size,
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              shuffle=False)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Train the model with early stopping
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=num_epochs,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size,
          callbacks=[early_stopping])

# Save the model
model.save('/home/priyanka/CI-CD/celeb_face.h5')