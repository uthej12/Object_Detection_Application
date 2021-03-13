import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2, numpy as np
import os
import pickle
from Models.vgg import VGG_16
from utils.handle_gpu_compatability import handle_gpu_compatibility
from utils.generate_pickle_files import pickle_generator
from utils.config import dataset_path

handle_gpu_compatibility()


labels = pickle_generator(dataset_path)

#load base model
baseModel = VGG_16('./trained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

#load pickle files
print('[INFO] loading pickle files...')
pickle_in = open("./pickle_files/X.pickle",'rb')
X = pickle.load(pickle_in)
X = np.array(X)
pickle_in = open("./pickle_files/y.pickle",'rb')
y = pickle.load(pickle_in)
y = np.array(y)

X = X.reshape(-1, 224, 224, 3)

#create train and test data
(trainX, testX, trainY, testY) = train_test_split(X, y,test_size=0.20, random_state=42)

#define the fully connected layer
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(4096, activation="relu", name='fc1')(headModel)
headModel = Dense(4096, activation="relu", name='fc2')(headModel)
headModel = Dense(20, activation="softmax", name='predictions')(headModel)

#freez the convolutionl layers and compile model
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

#compile model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

trainY = to_categorical(trainY)
testY = to_categorical(testY)

#train the model and save history
print('[INFO] started training model for 5 epochs...')
H = model.fit(trainX, trainY, batch_size=32, steps_per_epoch=len(trainX) // 32,
              validation_data=(testX, testY),
              validation_steps=len(testX) // 32,
              epochs=5)

#save the model
print("[INFO] saving the model ...")
model.save('./trained/custom_model.h5')
