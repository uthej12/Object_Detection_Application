import cv2
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
import numpy as np
import time
import random
import os
from utils.config import datadir,outdir

#define the annotation directory
annotation_dir = datadir+'/Annotations' 

#the files that have to be parsed through
file_name = ['train.txt','val.txt']

count =0
print('Loading Images') 
#iterate through each file at a time
for filename in file_name:
    #open the file and iterate through it line by line
    with open(datadir+'ImageSets/Main/'+filename) as file:
        for line in file:

            #the image name is the content of the line without the line break
            image_name = line.strip()
            path = datadir+'JPEGImages/'+line.strip()+'.jpg'
            #load the image using cv2
            img = cv2.imread(path, 1)
            
            #open the annotation file of the particular image and parse
            #through it using xml.etree
            doc = ET.parse(annotation_dir+'/'+image_name+'.xml').getroot()
            element_objs = doc.findall('object')

            #find all the objects present in the image
            for element_obj in element_objs:
                #get the class of the object
                class_name = element_obj.find('name').text

                #find the bounding box coordinates of the image
                obj_bbox = element_obj.find('bndbox')
                xmin = int(round(float(obj_bbox.find('xmin').text)))
                ymin = int(round(float(obj_bbox.find('ymin').text)))
                xmax = int(round(float(obj_bbox.find('xmax').text)))
                ymax = int(round(float(obj_bbox.find('ymax').text)))

                #crop the image and get the image of the object
                class_img = img[ymin:ymax,xmin:xmax]
                #resize the image to 224x224 
                class_img = cv2.resize(class_img,(224,224))
                #write it to the output directory
                #if output directory not present it will be created
                out_path = outdir+class_name
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                cv2.imwrite(out_path+'/'+image_name+'.jpg',class_img)
            count+=1

print('Completed ', count)