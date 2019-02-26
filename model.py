import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import pandas as pd
import PIL 
import csv
from sklearn.model_selection import *
from sklearn.utils import shuffle
from keras.layers import *
from keras.models import *
from keras.layers.normalization import *

width, height=320, 160
amendment=0.2 #the correction for right and left cemera
samples=[]
with open('mydata/driving_log.csv', 'r') as f:
    reader=csv.reader(f)
    for line in reader:
        samples.append(line) 
    del samples[0]
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    n_len=len(samples)
    while True:
        shuffle(samples)
        for i in range(0, n_len, batch_size):
            X=[]
            y=[]
            
            for row in samples[i:i+batch_size]:
                steering_center=float(row[3])#the third row is steering angle
                steering_left=steering_center+amendment
                steering_right=steering_center-amendment
                
                source_pathC =row[0] 
                tokensC = source_pathC.split('/')  
                filenameC = tokensC[-1]    
                local_pathC = 'mydata/IMG/' + filenameC   
                img_center=np.asarray(PIL.Image.open(local_pathC),dtype=np.uint8)
                
                
                source_pathL =row[1] 
                tokensL = source_pathL.split('/')  
                filenameL = tokensL[-1]    
                local_pathL = 'mydata/IMG/' + filenameL   
                img_left=np.asarray(PIL.Image.open(local_pathL),dtype=np.uint8)
                
                source_pathR =row[2] 
                tokensR = source_pathR.split('/')  
                filenameR = tokensR[-1]    
                local_pathR = 'mydata/IMG/' + filenameR  
                img_right=np.asarray(PIL.Image.open(local_pathR),dtype=np.uint8)
                                              
                img_flip=np.fliplr(img_center)# data augmentation with flip the image
                
                X.extend([img_center, img_left, img_right,img_flip])
                y.extend([steering_center, steering_left, steering_right,-steering_center])
               
            X=np.array(X)
            y=np.array(y)
            yield shuffle(X, y)

model=Sequential()

model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(height, width ,3)))#cropping the image 
model.add(Lambda(lambda x: x/255.0-0.5))#normalizing the iamges pixels' value

model.add(Conv2D(filters=16, kernel_size=3, strides=2, activation='relu',input_shape=(height, width, 1)))#43X159X16
model.add(Conv2D(filters=16, kernel_size=3, strides=1, activation=None))#42X157X16
model.add(BatchNormalization())#42X157X16
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))#21X79X16
model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))#19X77X32
model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation=None))#17X75X32
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))#9X38X32
model.add(Dropout(0.5))


model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))#7X36X32
model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation=None))#5X34X32
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))#3X17X32
model.add(Dropout(0.5))

model.add(Flatten())#1632

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1, activation=None))
model.compile(optimizer='adam', loss='mse')

train_generator=generator(train_samples, 32)
valid_generator=generator(validation_samples, 32)


model.fit_generator(train_generator, steps_per_epoch=1000, epochs=8,
                    validation_data=valid_generator, validation_steps=100)
 
model.summary()
model.save('mymodel.h5')
