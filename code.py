!pip install tensorflow gtts pytesseract opencv-python
!apt-get install -y libportaudio2 tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-mal
!pip install langdetect

import tensorflow as tf
import numpy as np
import cv2
import pytesseract
from google.colab import files
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from gtts import gTTS
from langdetect import detect
from IPython.display import Audio, display
import os

# Load Pre-trained MobileNetV2 Model for Object Detection
model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    """Loads an image and classifies it using MobileNetV2."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess for MobileNetV2

    # Predict the object in the image
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    object_name = decoded_predictions[0][1]  # Extract object name
    confidence = decoded_predictions[0][2]  # Extract confidence score

    return object_name, confidence


def extract_text(image_path):
    """Extracts text from an image using OCR (supports English, Hindi, and Malayalam)."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OCR to extract text in multiple languages
    text = pytesseract.image_to_string(gray, lang="eng+hin+mal").strip()

    if text:
        try:
            detected_lang = detect(text)  # Automatically detect language
        except:
            detected_lang = "en"  # Default to English if detection fails
    else:
        detected_lang = None

    return text if text else None, detected_lang

# Upload an image
uploaded = files.upload()

# Process the uploaded image
for filename in uploaded.keys():
    detected_text, language = extract_text(filename)

    if detected_text:
        print(f"Extracted Text: {detected_text} (Language: {language})")

        # Set language code for gTTS
        lang_code = "en"  # Default
        if language == "hi":
            lang_code = "hi"
        elif language == "ml":
            lang_code = "ml"

        # Convert detected text to speech
        tts = gTTS(detected_text, lang=lang_code)
        audio_file = "response.mp3"
        tts.save(audio_file)

        # Play the response audio
        display(Audio(audio_file, autoplay=True))
    else:
        object_name, confidence = classify_image(filename)
        print(f"Detected Object: {object_name} (Confidence: {confidence:.2f})")
