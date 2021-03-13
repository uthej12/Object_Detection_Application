import cv2, numpy as np
import os
import random
import pickle
from utils.config import pickle_file_path

def pickle_generator(path):

    #This function takes the path of the dataset as a argument

    #loading the labels of the dataset
    labels = os.listdir(path)

    #creating traning data
    training_data = []
    for label in labels:
        #iterating through each folder in the data directory
        class_num = labels.index(label)
        folder = os.path.join(path,label)
        print('Parsing images from ', folder)
        images = os.listdir(folder)
        #iterating through all the images in the directory
        for image in images:
            try:
                #reading the image and appending it to the training data 
                #as well as the class of the image
                img = cv2.imread(os.path.join(folder,image))
                training_data.append([img,class_num])
            except:
                pass

    #shuffle the training data
    random.shuffle(training_data)

    X = []
    y = []

    #split the training data into data and labels
    #which is X and y respectively
    for features,label in training_data:
        X.append(features)
        y.append(label)

    #save the data as X.pickle
    pickle_out = open(pickle_file_path+"X.pickle",'wb')
    pickle.dump(X,pickle_out)
    pickle_out.close()

    #save the labels as y.pickle
    pickle_out = open(pickle_file_path+"/y.pickle",'wb')
    pickle.dump(y,pickle_out)
    pickle_out.close()

    #return the labels
    return labels

pickle_generator()
    
