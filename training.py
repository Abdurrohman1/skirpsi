from google.colab import drive
drive.mount('/content/drive/')
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
from glob import glob
import random
from tensorflow.keras import layers, models, Input, callbacks
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import ImageDataGenerator
train_dir = '/content/drive/MyDrive/ Abdurrohman - Brain Tumor VGG16/Data/Training'
test_dir = '/content/drive/MyDrive/ Abdurrohman - Brain Tumor VGG16/Data/Testing'
IMG_SIZE = (224, 224)

datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.15
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size = IMG_SIZE,
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training',
    shuffle = True)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size = IMG_SIZE,
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = True)


test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size = IMG_SIZE,
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True
)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 4 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint('brain_tumor_vgg16.h5',
                            monitor='val_accuracy',
                            save_best_only=True,
                            mode='max',
                            verbose=1)

early_stopping = EarlyStopping(monitor='val_loss',
                              patience=5,
                              restore_best_weights=True)

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,
    epochs=30,
    callbacks=[checkpoint, early_stopping]
)

model.save('/content/drive/MyDrive/ Abdurrohman - Brain Tumor VGG16/Model/brain_tumor_model.h5')
# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg

def predict_tumor(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    # Get class label
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Display the image
    plt.imshow(mpimg.imread(img_path))
    plt.axis('off')
    plt.title(f'Predicted: {class_labels[class_idx]}\nConfidence: {confidence:.2f}')
    plt.show()

    return class_labels[class_idx], confidence

