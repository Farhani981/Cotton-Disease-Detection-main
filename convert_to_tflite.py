import tensorflow as tf
import os

print("Loading model...")
model = tf.keras.models.load_model('resnet50.h5')

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("Saving TFLite model...")
with open('resnet50.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! resnet50.tflite has been created.")
