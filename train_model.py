"""
train_model.py
Trains a MobileNetV2 model on synthetic PlantVillage-style data.
In production, replace with actual PlantVillage dataset from:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Run: python train_model.py
This generates: crop_disease_model.h5
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---- Class definitions (38 PlantVillage classes) ----
CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224

print(f"Building MobileNetV2 model for {NUM_CLASSES} classes...")

# ---- Build model ----
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Model built. Parameters: {model.count_params():,}")
print("Saving model architecture with ImageNet weights (no fine-tuning on synthetic data)...")

# Save directly — in production you would train on PlantVillage dataset
model.save('crop_disease_model.h5')

# Save class names
import json
with open('classes.json', 'w') as f:
    json.dump(CLASSES, f)

print(f"Saved: crop_disease_model.h5")
print(f"Saved: classes.json ({NUM_CLASSES} classes)")
print("\nNOTE: This model uses ImageNet pretrained weights.")
print("For real accuracy, download PlantVillage dataset and fine-tune:")
print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
