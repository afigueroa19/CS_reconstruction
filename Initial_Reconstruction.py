# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:18:52 2022

@author: Alberto Figueroa
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import time
from keras.layers import Input, Conv2D, Lambda, Reshape, Multiply, Add, Subtract
from keras.activations import relu
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.utils import plot_model

# x_test = np.load('x_test.npy' )
# x_train = np.load('x_train.npy')

# measure=50 # Las unicas opciones son 50 y 10
# Type=1  #0 para binaria, 1 para gaussiana
# if measure==10:
#     if Type==0:
#         y_test = np.load('y_test_10.npy')    
#         y_train = np.load('y_train_10.npy')
#         print("10% de medicion Binario")
#     elif Type==1:
#         y_test = np.load('y_test_10_Gauss.npy')    
#         y_train = np.load('y_train_10_Gauss.npy')
#         print("10% de medicion Gaussiano")
# elif measure==50:
#     if Type==0:
#         y_test = np.load('y_test_50.npy')    
#         y_train = np.load('y_train_50.npy')
#         print("50% de medicion Binario")
#     elif Type==1:
#         y_test = np.load('y_test_50_Gauss.npy')    
#         y_train = np.load('y_train_50_Gauss.npy')
#         print("50% de medicion Gauss")

def Initial_Recon(y_train, x_train, i):
    
    
    name_model = 'Initial_Recon_88912_Bi50_V{i}.h5'
    
    csv_name= 'Initial_Recon_88912_Bi50_V{i}.log'
    
    inp = tf.keras.Input(shape=(545,))
    
    Dense = tf.keras.layers.Dense(1089, activation="relu", use_bias=False, kernel_initializer="random_normal")(inp)
    
    # Dense2 = tf.keras.layers.Dense(1089, activation="relu", use_bias=False, kernel_initializer="random_normal")(Dense1)
    
    # output = tf.keras.layers.Reshape((33, 33), name='output')(Dense)
    
    model = tf.keras.Model(inputs=[inp] , outputs=[Dense], name=name_model)

    
    model.summary()
    
    
    model.compile(optimizer= Adam(learning_rate=0.0001),\
                    metrics='accuracy',  loss="mse")
    
    
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.00001)
    
    
    
    csv_logger = CSVLogger(csv_name , separator=',', append=False) 
    
    # history = model.fit(x = y_train, y = x_train, epochs=300, validation_data=(y_test, x_test),
    #                 batch_size = 64, shuffle=True, callbacks=[csv_logger])
    history = model.fit(x = y_train, y = x_train, epochs=300,
                    batch_size = 64, shuffle=True, callbacks=[csv_logger])
        
    model.save(name_model) 
    
    # model.save_weights("RECON_300epochs_weights.h5")
    
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('ISTAxRECON Reconstruction Bi')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('ISTAxRECON Reconstruction Bi')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
    
    return model, history, csv_logger
