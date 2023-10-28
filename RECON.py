# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 08:11:34 2022

@author: afigu
RECON
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import time
from keras.layers import Input, Conv2D, Lambda, Reshape, Multiply, Add, Subtract
from keras.activations import relu
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.utils import plot_model


def RECON(y_train, x_train, i):

        
    name_model = f'RECON_NET_88912_Binary01_V{i}].h5'
    
    csv_name= f'RECON_NET_88912_Binary01_V{i}.log'
    
    
    
    inp = tf.keras.Input(shape=(1089,))
    
    conv1_x2 = tf.keras.layers.Reshape((33, 33, 1), name='conv1_x2')(inp)
    
    conv1_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv1_x3', kernel_initializer="glorot_normal")(conv1_x2)
    
    conv1_x4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv1_x4', kernel_initializer="glorot_normal")(conv1_x3) #Paso intermedio
    
    conv1_x5 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv1_x5', kernel_initializer="glorot_normal")(conv1_x4)
    
    output = tf.keras.layers.Reshape((1089,), name='output')(conv1_x5)
    
    model = tf.keras.Model(inputs=[inp] , outputs=[output], name=name_model)
    
    model.summary()
    
    
    model.compile(optimizer= Adam(learning_rate=0.0001),\
                            metrics='accuracy',  loss="mse")
    
    
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.00001)
    
    csv_logger = CSVLogger(csv_name, separator=',', append=False)
    
    history = model.fit(x = y_train, y = x_train, epochs=200,
                    batch_size = 64, shuffle=True, callbacks=[csv_logger])
    
    
    # print(history.history)
    
    model.save(name_model)
    
    # model.save_weights("RECON_300epochs_weights.h5")
    
    # plt.plot(history.history['loss'])
    # # plt.plot(history.history['val_loss'])
    # plt.title('RECON-NET_Gauss Reconstruction')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    # plt.plot(history.history['accuracy'])
    # # plt.plot(history.history['val_accuracy'])
    # plt.title('RECON-NET_Gauss Reconstruction')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    
    return model, history, csv_logger

