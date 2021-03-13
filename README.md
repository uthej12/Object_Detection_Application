# Object_Detection_Application

This application is a basic object detection application created using tensorflow and OpenCV. A basic approach to Object detection has been taken in this application.

The steps to run the application are given below.
1. Tensorflow GPU is required to run the application.

2. Can be run in colab with modificatations in the code commented in the code. The web application will not work in colab only the script will.

3. Requirements
This application was developed and installed in anaconda environment and a virtual environment was craeted and the floowing packages were installed

tensorflow >2.1(GPU)
opencv
scikit learn
numpy

4. The config file contains the specific files. This file has to be eddited in order for the application to work according to the users system.

5.In case of web app just run python main.py. Web app was created using Flask.

6.In colab the inference script should be run. path of the image for prediction should be provided in the script.

7.Model can be trained using the fine_tune.py script for this the dataset_generator.py script should be run on the VOC2012 dataset and the dataset should be stored in appropriate location and should be updated in config.
link for VOC2012 dataset
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

8. A folder named trained should be created in the root directory where the weights of a pretrained vgg16 model will be saved.

9. The fine tuning will be done and the model will be stored in trained folder which already contains the vgg16 pretrained model.

10. In case of fine tuning the dataset must be downloaded and both generate_dataset and generate pickle files must be executed

