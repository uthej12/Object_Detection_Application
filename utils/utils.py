import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def selective_search(image, method="fast"):
    # initialize OpenCV's selective search implementation and set the
    # input image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    # check to see if we are using the *fast* but *less accurate* version
    # of selective search
    if method == "fast":
        ss.switchToSelectiveSearchFast()
    # otherwise we are using the *slower* but *more accurate* version
    else:
        ss.switchToSelectiveSearchQuality()
    # run selective search on the input image
    rects = ss.process()
    # return the region proposal bounding boxes
    return rects


def arrange_results(proba,boxes,labels):
    result = []
    index = 0 
    new_index = 0
    #iterate through all the probabilities of each box
    for res in proba:
        temp = {}
        #if the probability greater than 0.99
        if max(res) > 0.99:
            max_prob = max(res)
            #get the class of the prediction using argmax
            prediction = labels[np.argmax(res)]
            #create a dictionary to store all the values
            temp['class'] = prediction
            temp['probability'] = max_prob
            temp['box'] = boxes[index]
            #add the dictionary to the result
            result.append(temp)
            new_index +=1
        index +=1
        #return the result
    return result

def handle_gpu_compatibility():
    """Handling of GPU issues for cuDNN initialize error and memory issues."""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)


