#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add
from tensorflow.keras.utils import to_categorical


model = load_model('./model_weights/model_9.h5')





modelTemp = ResNet50(weights='imagenet', input_shape=(224,224,3))





modelResnet = Model(modelTemp.input, modelTemp.layers[-2].output)




def preprocessImage(img):
    img = image.load_img(img,  target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img



def encodeImage(img):
    img = preprocessImage(img)
    featureVector = modelResnet.predict(img)
    featureVector = featureVector.reshape(1, featureVector.shape[1])
    return featureVector





import pickle
with open('./wordToIdx.pkl', 'rb') as w2i:
    wordToIdx = pickle.load(w2i)
with open('./idxToWord.pkl', 'rb') as i2w:
    idxToWord = pickle.load(i2w)





def predictCaptions(photo):
    
    inText = "<s>"
    maxLen = 38
    for i in range(maxLen):
        sequence = [wordToIdx[w] for w in inText.split() if w in wordToIdx]
        sequence = pad_sequences([sequence], maxlen = maxLen, padding='post')
        
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax() # word with max prob always - greedy sampling
        word = idxToWord[ypred]
        
        inText += " " + word
        if word == '<e>':
            break
    finalCaption = inText.split()[1:-1]
    finalCaption = " ".join(finalCaption)
    
    return finalCaption





def captionTheImg(image):
    enc = encodeImage(image)
    captions = predictCaptions(enc)
    return captions







