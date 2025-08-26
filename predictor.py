import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


MODEL_PATH = os.path.join('model','brain_tumor_model.h5')

model = load_model(MODEL_PATH)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    return predicted_class, confidence
