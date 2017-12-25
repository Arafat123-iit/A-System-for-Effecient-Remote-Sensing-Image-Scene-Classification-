 Remote Sensing Image Scene Classification Using Deep Learning
 
 Here we classified scene images of NWPU-RESISC45 dataset
 The dataset can be downloaded from http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html
 We applied Convolutional Neural Network on this dataset to classifiy images. We trained the dataset from scratch. Also appied transfer learning using pre-trained VGG16 abd ResNet50.
 
 We used training_from_scratch.py to train the dataset from scratch. We used python 2.7, keras 2.05 and Theano backend 1.0,OpenCV 3.30.
 Applied transfer learning using VGG16_Transfer_Learning.py and RestNet50_Transfer_Learning.py. Here we used python 3.5, keras, 2.05 using Tensorfloe backend 1.0 and openCV 3.30 
