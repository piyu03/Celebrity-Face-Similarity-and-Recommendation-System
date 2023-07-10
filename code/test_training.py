import pytest
from training import *

def test_model_training():
    # Define the paths to the training and validation data for the unit test
    base_dir = '/home/priyanka/CI-CD/images_split'
    train_dir = os.path.join(base_dir, 'aug_train_folder')
    validation_dir = os.path.join(base_dir, 'valid_folder')

    # Define the parameters for the unit test
    image_size = (224, 224)
    batch_size = 16
    num_epochs = 2

    # Preprocess the data using data augmentation for training and rescaling for validation
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training and validation data for the unit test
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

    # Train the model with a small number of epochs for the unit test
    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              epochs=num_epochs,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // batch_size,
              verbose=0)

    # Perform assertions to validate the model training
    assert model.history is not None
    assert 'accuracy' in model.history.history
    assert 'val_accuracy' in model.history.history
    assert len(model.history.history['accuracy']) == num_epochs
    assert len(model.history.history['val_accuracy']) == num_epochs

    # Add additional assertions based on your specific requirements

    # Clean up any resources if needed

# Run the unit test
pytest.main(['-v', 'your_test_script.py'])
