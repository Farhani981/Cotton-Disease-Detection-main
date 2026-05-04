import os
import numpy as np
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    print("TensorFlow imported successfully.")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    exit(1)

MODEL_PATH = 'resnet50.h5'

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    exit(1)

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # Test with dummy data
    print("Testing with dummy data...")
    dummy_input = np.random.rand(1, 224, 224, 3)
    prediction = model.predict(dummy_input)
    print(f"Prediction result: {prediction}")
    print(f"Predicted class: {np.argmax(prediction)}")
    
except Exception as e:
    print(f"Error during model operations: {e}")
