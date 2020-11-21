from config import *
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

data = []
labels = []
bboxes = []
imagePaths = []

for file in os.listdir(ANNOTS_PATH):
  rows = open(os.path.join(ANNOTS_PATH,file)).read().strip().split("\n")
  for row in rows[1:]:
    row = row.split(",")
    (_, filename, startX, startY, endX, endY, label) = row
    imagePath = os.path.sep.join([IMAGES_PATH, label, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    data.append(image)
    labels.append(label)
    bboxes.append((startX, startY, endX, endY))
    imagePaths.append(imagePath)
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.2, random_state=0)
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]
f = open(TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

base_model = VGG16(include_top = False, weights = 'imagenet', input_shape=(224,224,3))
base_model.trainable = False
x = y = Flatten()(base_model.output)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.25)(x)
classification_preds = Dense(len(lb.classes_),name = 'class_label', activation = 'softmax')(x)

y = Dense(256, activation = 'relu')(y)
y = Dense(128, activation = 'relu')(y)
y = Dense(64, activation = 'relu')(y)
y = Dense(32, activation = 'relu')(y)
bbox_preds = Dense(4, name = 'bounding_box', activation = 'sigmoid')(y)

model = Model(inputs = base_model.input, outputs = (bbox_preds, classification_preds))

losses = {'class': 'categorical_crossentropy', 'bbox': 'mean_squared_error'}

lossWeights = {'class': 1.0, 'bbox': 1.0}

opt = Adam(lr = LR)

model.compile(loss=losses, optimizer = opt, metrics = ['accuracy'], loss_weights = lossWeights)

trainTargets = {'class': trainLabels, 'bbox': trainBBoxes}

testTargets = {'class': testLabels, 'bbox': testBBoxes}

trainer = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)
model.save(MODEL_PATH, save_format = 'h5')
f = open(LB_PATH, 'wb')
f.write(pickle.dumps(lb))
f.close()

