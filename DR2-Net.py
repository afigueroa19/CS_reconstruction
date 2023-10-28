# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 08:18:00 2022

@author: afigu

DR2-Net
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



# model, history, csv_logger = HSCNN_R(prediction_train50, x_train, 0)
# model, history, csv_logger = HSCNN_R(prediction_train10, x_train, 1)
# model, history, csv_logger = HSCNN_R(prediction_train01, x_train, 2)

# model, history, csv_logger = AMP(prediction_train50, x_train, 0)
# model, history, csv_logger = AMP(prediction_train10, x_train, 1)
# model, history, csv_logger = AMP(prediction_train01, x_train, 2)

# model, history, csv_logger = DR2(prediction_train50, x_train, 0)
# model, history, csv_logger = DR2(prediction_train10, x_train, 1)
# model, history, csv_logger = DR2(prediction_train01, x_train, 2)

# model, history, csv_logger = RECON(prediction_train50, x_train, 0)
# model, history, csv_logger = RECON(prediction_train10, x_train, 1)
# model, history, csv_logger = RECON(prediction_train01, x_train, 2)

# model, history, csv_logger = ISTAxRECON(prediction_train10, x_train, 1)
# model, history, csv_logger = ISTAxRECON(prediction_train01, x_train, 2)
def DR2(y_train, x_train, i):

        
    name_model = f'DR2_88912_Binary01_V{i}.h5'
    
    csv_name= f'DR2_88912_Binary01_V{i}.log'
    
    
    inp = tf.keras.Input(shape=(1089,))
    
    conv1_x2 = tf.keras.layers.Reshape((33, 33, 1), name='conv1_x2')(inp)
    
    conv1_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv1_x3', kernel_initializer="glorot_normal")(conv1_x2)
    
    conv1_x3 =tf.keras.layers.BatchNormalization()(conv1_x3)
    
    conv1_x4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv1_x4', kernel_initializer="glorot_normal")(conv1_x3) 
    
    conv1_x4 =tf.keras.layers.BatchNormalization()(conv1_x4)
    
    conv1_x5 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv1_x5', kernel_initializer="glorot_normal")(conv1_x4)
    
    conv1_x6 = tf.keras.layers.Add(name='conv1_x6')([conv1_x5, conv1_x2])
    
    """
    Block-2
    """
    
    conv2_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv2_x3', kernel_initializer="glorot_normal")(conv1_x6)
    
    conv2_x3 =tf.keras.layers.BatchNormalization()(conv2_x3)
    
    conv2_x4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv2_x4', kernel_initializer="glorot_normal")(conv2_x3) 
    
    conv2_x4 =tf.keras.layers.BatchNormalization()(conv2_x4)
    
    conv2_x5 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv2_x5', kernel_initializer="glorot_normal")(conv2_x4)
    
    conv2_x6 = tf.keras.layers.Add(name='conv2_x6')([conv2_x5, conv1_x6])
    
    """
    Block-3
    """
    
    conv3_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv3_x3', kernel_initializer="glorot_normal")(conv2_x6)
    
    conv3_x3 =tf.keras.layers.BatchNormalization()(conv3_x3)
    
    conv3_x4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv3_x4', kernel_initializer="glorot_normal")(conv3_x3) 
    
    conv3_x4 =tf.keras.layers.BatchNormalization()(conv3_x4)
    
    conv3_x5 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv3_x5', kernel_initializer="glorot_normal")(conv3_x4)
    
    conv3_x6 = tf.keras.layers.Add(name='conv3_x6')([conv3_x5, conv2_x6])
    
    """
    Block-4
    """
    
    conv4_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv4_x3', kernel_initializer="glorot_normal")(conv3_x6)
    
    conv4_x3 =tf.keras.layers.BatchNormalization()(conv4_x3)
    
    conv4_x4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv4_x4', kernel_initializer="glorot_normal")(conv4_x3) 
    
    conv4_x4 =tf.keras.layers.BatchNormalization()(conv4_x4)
    
    conv4_x5 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv4_x5', kernel_initializer="glorot_normal")(conv4_x4)
    
    conv4_x6 = tf.keras.layers.Add(name='conv4_x6')([conv4_x5, conv3_x6])
    
    output = tf.keras.layers.Reshape((1089,), name='output')(conv4_x6)
    
    model = tf.keras.Model(inputs=[inp] , outputs=[output], name=name_model)
    
    
    
    model.summary()
    
    
    
    model.compile(optimizer=Adam(learning_rate=0.0001),\
                        loss="mse",
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
    # plt.title('DR2_recon_Gauss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    # plt.plot(history.history['accuracy'])
    # # plt.plot(history.history['val_accuracy'])
    # plt.title('DR2_recon_Gauss')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    
    return model, history, csv_logger

# model, history, csv_logger = DR2(prediction_train, x_train, 0)
# model, history, csv_logger = AMP(prediction_train, x_train, 0)
