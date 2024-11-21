import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    return model

def prepare_image(img_path, img_height, img_width):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(model, img_array, threshold=0.5):
    prediction = model.predict(img_array)[0][0]
    label = 'gf' if prediction > threshold else 'me'
    confidence = float(prediction) if label == 'gf' else float(1 - prediction)
    return label, confidence

if __name__ == "__main__":
    image_path = 'test_images/09092008941.jpg'
    model_path = 'models/best_model.keras'

    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        exit(1)

    model = load_trained_model(model_path)
    img_array = prepare_image(image_path, 224, 224)
    label, confidence = predict_image(model, img_array)

    print(f'Predicted Label: {label} with confidence {confidence:.2f}')
