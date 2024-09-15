import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import csv

TRAIN_DIR_1 = 'train/autistic'
TRAIN_DIR_2 = 'train/non_autistic'
TEST_DIR = 'test'
IMG_SIZE = 150
LR = 1e-3
MODEL_NAME = 'Detect_Autism'

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR_1)):
        path = os.path.join(TRAIN_DIR_1, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])

    for img in tqdm(os.listdir(TRAIN_DIR_2)):
        path = os.path.join(TRAIN_DIR_2, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append(np.array(img_data))

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

def create_label(img_name):
    label = img_name.split('(')
    if label[0] == 'autistic ':
        return np.array([1, 0])
    elif label[0] == 'non_autistic ':
        return np.array([0, 1])

# Load or create data
if os.path.exists('train_data.npy'):
    train_data = np.load('train_data.npy', allow_pickle=True)
else:
    train_data = create_train_data()

if os.path.exists('test_data.npy'):
    test_data = np.load('test_data.npy', allow_pickle=True)
else:
    test_data = create_test_data()

# Prepare training and testing data
X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train_data])
X_test = np.array([i for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
if os.path.exists('model.h5'):
    model.load_weights('model.h5')
else:
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1)
    model.save('model.h5')

# Predict on test data
results = []
for img in os.listdir(TEST_DIR):
    image = cv2.imread(os.path.join(TEST_DIR, img), 0)
    test_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(test_img)[0]
    label = 'autistic' if prediction[0] > prediction[1] else 'non_autistic'
    results.append([img, label])

# Save results to CSV
with open('Submit.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Prediction'])
    writer.writerows(results)
