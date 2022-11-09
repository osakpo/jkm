# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:27:28 2022

@author: Po
"""

import keras
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, weights_file):
    #load the model
    
    model = keras.models.load_model('keras_Model.h5', compile=False)
    
    #create the array of the righ shape to feed into th keras model
    data = np.ndarray(shape = (1, 224, 224, 3), dtype=np.float32)
    image = img
    #image string
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
   # turn the image into an array
    image_array = np.asarray(image)
   #Normalize the image
    normalized_image_array = (image_array.astype(np.float32)/ 127.0) - 1
   
   #load the image into the array
    data[0] = normalized_image_array
   
   #run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) #return position of the highest probability