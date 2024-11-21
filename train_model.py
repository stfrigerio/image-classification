import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.utils import plot_history, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Define paths
dataset_dir = 'dataset'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 20

# Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Building the Model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True)

# Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stop, checkpoint]
)

# Fine-Tuning (Optional)
# Unfreeze the last 10 layers of the base model
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)

# Plot Training History
plot_history(history, history_fine)

# Load the Best Model
best_model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model.keras'))

# Evaluate on Validation Data
loss, accuracy = best_model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy*100:.2f}%')

# Confusion Matrix
validation_generator.reset()
Y_true = validation_generator.classes
Y_pred = best_model.predict(validation_generator)
Y_pred = np.where(Y_pred > 0.5, 1, 0).flatten()

cm = confusion_matrix(Y_true, Y_pred)
print('Confusion Matrix')
print(cm)

# Plot and Save Confusion Matrix
plot_confusion_matrix(Y_true, Y_pred, classes=list(validation_generator.class_indices.keys()))

# Save the Final Model
best_model.save(os.path.join(model_dir, 'face_classifier_model.keras'))
print("Model saved to models/face_classifier_model.keras")
