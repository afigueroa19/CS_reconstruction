# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:57:38 2022

@author: Alberto Figueroa
HSCNN-R
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


def HSCNN_R(y_train, x_train, i):

    name_model = f'HSCNN-R_88912_Binary01_V{i}.h5'
    
    csv_name= f'HSCNN-R_88912_Binary01_V{i}.log'
    
    inp = tf.keras.Input(shape=(1089,))
    
    conv0_x1 = tf.keras.layers.Reshape((33, 33, 1), name='conv0_x1')(inp)
    
    conv0_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv0_x2', kernel_initializer="glorot_normal")(conv0_x1)
    
    """
    Block-1
    """
    
    conv1_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv1_x1', kernel_initializer="glorot_normal")(conv0_x2) 
    
    conv1_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv1_x2', kernel_initializer="glorot_normal")(conv1_x1)
    
    conv1_x3 = tf.keras.layers.Add(name='conv1_x3')([conv0_x2, conv1_x2])
    
    conv1 = tf.keras.layers.ReLU()(conv1_x3)
    
    """
    Block-2
    """
    
    conv2_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv2_x1', kernel_initializer="glorot_normal")(conv1) 
    
    conv2_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv2_x2', kernel_initializer="glorot_normal")(conv2_x1)
    
    conv2_x3 = tf.keras.layers.Add(name='conv2_x3')([conv1, conv2_x2])
    
    conv2 = tf.keras.layers.ReLU()(conv2_x3)
    
    """
    Block-3
    """
    
    conv3_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv3_x1', kernel_initializer="glorot_normal")(conv2) 
    
    conv3_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv3_x2', kernel_initializer="glorot_normal")(conv3_x1)
    
    conv3_x3 = tf.keras.layers.Add(name='conv3_x3')([conv2, conv3_x2])
    
    conv3 = tf.keras.layers.ReLU()(conv3_x3)
    
    """
    Block-4
    """
    
    conv4_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv4_x1', kernel_initializer="glorot_normal")(conv3) 
    
    conv4_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv4_x2', kernel_initializer="glorot_normal")(conv4_x1)
    
    conv4_x3 = tf.keras.layers.Add(name='conv4_x3')([conv3, conv4_x2])
    
    conv4 = tf.keras.layers.ReLU()(conv4_x3)
    
    """
    Block-5
    """
    
    conv5_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv5_x1', kernel_initializer="glorot_normal")(conv4) 
    
    conv5_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv5_x2', kernel_initializer="glorot_normal")(conv5_x1)
    
    conv5_x3 = tf.keras.layers.Add(name='conv5_x3')([conv4, conv5_x2])
    
    conv5 = tf.keras.layers.ReLU()(conv5_x3)
    
    """
    Block-6
    """
    
    conv6_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv6_x1', kernel_initializer="glorot_normal")(conv5) 
    
    conv6_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv6_x2', kernel_initializer="glorot_normal")(conv6_x1)
    
    conv6_x3 = tf.keras.layers.Add(name='conv6_x3')([conv5, conv6_x2])
    
    conv6 = tf.keras.layers.ReLU()(conv6_x3)
    
    """
    Block-7
    """
    
    conv7_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv7_x1', kernel_initializer="glorot_normal")(conv6) 
    
    conv7_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv7_x2', kernel_initializer="glorot_normal")(conv7_x1)
    
    conv7_x3 = tf.keras.layers.Add(name='conv7_x3')([conv6, conv7_x2])
    
    conv7 = tf.keras.layers.ReLU()(conv7_x3)
    
    """
    Block-8
    """
    
    conv8_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv8_x1', kernel_initializer="glorot_normal")(conv7) 
    
    conv8_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv8_x2', kernel_initializer="glorot_normal")(conv8_x1)
    
    conv8_x3 = tf.keras.layers.Add(name='conv8_x3')([conv7, conv8_x2])
    
    conv8 = tf.keras.layers.ReLU()(conv8_x3)
    
    """
    Block-9
    """
    
    conv9_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv9_x1', kernel_initializer="glorot_normal")(conv8) 
    
    conv9_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv9_x2', kernel_initializer="glorot_normal")(conv9_x1)
    
    conv9_x3 = tf.keras.layers.Add(name='conv9_x3')([conv8, conv9_x2])
    
    conv9 = tf.keras.layers.ReLU()(conv9_x3)

    """
    Block-10
    """
    
    conv10_x1 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, activation='ReLU', name='conv10_x1', kernel_initializer="glorot_normal")(conv9) 
    
    conv10_x2 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='conv10_x2', kernel_initializer="glorot_normal")(conv10_x1)
    
    conv10_x3 = tf.keras.layers.Add(name='conv10_x3')([conv9, conv10_x2])
    
    conv10 = tf.keras.layers.ReLU()(conv10_x3)
    
    """
    Output
    """
    
    out1 = tf.keras.layers.Add(name='out1')([conv10, conv0_x2])
    
    out2 = tf.keras.layers.ReLU()(out1)
    
    out3 = tf.keras.layers.Conv2D(64, [3,3], padding='SAME', use_bias=False, name='out3', kernel_initializer="glorot_normal")(out2)
    
    out4 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False, name='out4', kernel_initializer="glorot_normal")(out3)
    
    output = tf.keras.layers.Reshape((1089,), name='output')(out4)

    

    model = tf.keras.Model(inputs=[inp] , outputs=[output], name=name_model)
    
    
    
    model.summary()
    
    
    
    model.compile(optimizer=Adam(learning_rate=0.0001),\
                        loss=tf.keras.losses.MeanSquaredError(),\
                        metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, to_file='model_plot.png')  
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=13, min_delta=0.00001)
    
    csv_logger = CSVLogger(csv_name, separator=',', append=False)
    
    history = model.fit(y_train, x_train, epochs=200,
                    batch_size = 64, shuffle=True, callbacks=[csv_logger])
    
    # history = model.fit(y_train, x_train, epochs=50, validation_data=(y_test, x_test),
    #                 batch_size = 64, shuffle=True, callbacks=[callback, csv_logger])
    
    
    
    model.save(name_model)
    
    # plt.plot(history.history['loss'])
    # # plt.plot(history.history['val_loss'])
    # plt.title('HSCNN-R Gauss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    # plt.plot(history.history['accuracy'])
    # # plt.plot(history.history['val_accuracy'])
    # plt.title('HSCNN-R Gauss')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    
    return model, history, csv_logger
