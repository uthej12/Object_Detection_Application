import cv2
import tensorflow as tf
import os
import numpy as np
from Models.vgg import VGG_16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import matplotlib.pyplot as plt
from utils.utils import resize
from utils.utils import selective_search
from utils.utils import arrange_results
from utils.nms import non_max_suppression
from utils.utils import handle_gpu_compatibility
import time 
from utils.config import labels


handle_gpu_compatibility()

def predict(image):
    
    #load the base model and add the FC layer
    print('[INFO] Loading Model')
    baseModel = VGG_16('./trained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(4096, activation="relu", name='fc1')(headModel)
    headModel = Dense(4096, activation="relu", name='fc2')(headModel)
    headModel = Dense(20, activation="softmax", name='predictions')(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.load_weights('./trained/custom_model.h5') 
    print('[INFO] Model Loaded Sucessfully')

    print('[INFO] Loading Image')
    image = resize(image,height=500,width=500)

    print('[INFO] Running Selective Search on the image')
    start = time.time()
    rects = selective_search(image)
    end = time.time()
    print('[INFO] Selective search took.{} s'.format(end-start))

    clone = image.copy()
    for (startX, startY, endX, endY) in rects:
        # draw the bounding box and label on the image
        color =  (np.random.randint(255, size=3))
        color= (int(color[0]),int(color[1]),int(color[2]))
        cv2.rectangle(clone, (startX, startY), (endX, endY), color, 1)

    stage1 = clone

    proposals = []
    boxes = []
    #filtering out small region proposals
    for (x, y, w, h) in rects[:2000]:

        if w / float(image.shape[0]) < 0.1 or h / float(image.shape[0]) < 0.1:
            continue
        roi = image[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (224,224),
            interpolation=cv2.INTER_CUBIC)
        
        roi = np.array(roi)
        roi = np.reshape(roi,(224,224,3))
        
        proposals.append(roi)
        boxes.append((x, y, x + w, y + h))

    proposals = np.array(proposals, dtype="float32")
    boxes = np.array(boxes, dtype="int32")
    print("Selective search extracted {} regions".format(proposals.shape[0]))

    clone = image.copy()
    for (startX, startY, endX, endY) in boxes:
        # draw the bounding box and label on the image
        color =  (np.random.randint(255, size=3))
        color= (int(color[0]),int(color[1]),int(color[2]))
        cv2.rectangle(clone, (startX, startY), (endX, endY), color, 1)

    stage2 = clone

    print('[INFO] Predicting Proposals for extracted regions')
    start = time.time()
    proba = model.predict(proposals)
    end = time.time()
    print('[INFO] Prediction took.{} s'.format(end-start))

    result = arrange_results(proba,boxes,labels)

    sorted_results = sorted(result, key=lambda x: x['probability'], reverse=True)
    print("[INFO] Number of accepted proposals",len(sorted_results))

    final_rects = []
    final_probablities = []
    final_labels = []
    split= int(len(sorted_results) * 0.45)
    for value in sorted_results[:split]:
        final_rects.append(value['box'])
        final_probablities.append(value['probability'])
        final_labels.append(value['class'])

    boxes = final_rects
    clone = image.copy()
    index_gen = [x.tolist() for x in final_rects]
    for (startX, startY, endX, endY) in boxes:
        # draw the bounding box and label on the image
        label = final_labels[index_gen.index([startX, startY, endX, endY])]
        color =  (np.random.randint(255, size=3))
        color= (int(color[0]),int(color[1]),int(color[2]))
        cv2.rectangle(clone, (startX, startY), (endX, endY),
            color, 1)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    stage3 = clone

    #apply non max suppression
    boxes = non_max_suppression(final_rects, final_probablities)

    clone = image.copy()
    index_gen = [x.tolist() for x in final_rects]
    for (startX, startY, endX, endY) in boxes:
        label = final_labels[index_gen.index([startX, startY, endX, endY])]
        cv2.rectangle(clone, (startX, startY), (endX, endY),
            (0, 255, 0), 1)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    stage4 = clone

    return stage1,stage2,stage3,stage4

if __name__ == "__main__":
    image_path = 'C:\\Users\\Uthej Goud\\Documents\\Work\\Thesis\\tests\\airplane.jpg' #the path of the image to run prediction on
    image = cv2.imread(image_path)
    stage1,stage2,stag3,result = predict(image)
    cv2.imshow('result',result)
    cv2.waitKey(0)

