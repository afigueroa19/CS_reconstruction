# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:05:38 2022

@author: Alberto Figueroa

AMP-Net
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



def AMP(y_train, x_train, i):


    name_model = 'AMP_88912_Binary01_V{i}.h5'
    
    csv_name= 'AMP_88912_Binary01_V{i}.log'
    
    
    inp = tf.keras.Input(shape=(1089,))
    
    conv0_x1 = tf.keras.layers.Reshape((33, 33, 1), name='conv0_x1')(inp)
    
    """
    Block-1
    """
    
    conv1_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv1_x1', kernel_initializer="glorot_normal")(conv0_x1) 
    
    conv1_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv1_x2', kernel_initializer="glorot_normal")(conv1_x1)
    
    conv1_x3 =tf.keras.layers.BatchNormalization()(conv1_x2)
    
    conv1_x4 =tf.keras.layers.ReLU()(conv1_x3)   
    
    conv1_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv1_x5', kernel_initializer="glorot_normal")(conv1_x4)
    
    conv1_x6 =tf.keras.layers.BatchNormalization()(conv1_x5)      

    conv1_x7 =tf.keras.layers.ReLU()(conv1_x6)     

    conv1_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv1_x8', kernel_initializer="glorot_normal")(conv1_x7)
    
    conv1_x9 =tf.keras.layers.BatchNormalization()(conv1_x8)
    
    conv1_x10 =tf.keras.layers.ReLU()(conv1_x9)   
    
    conv1_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv1_x11', kernel_initializer="glorot_normal")(conv1_x10)        

    conv1_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv1_x12', kernel_initializer="glorot_normal")(conv1_x11)                                                                                                               
    
    conv1 = tf.keras.layers.Add(name='conv1')([conv1_x12, conv0_x1])
    
    """
    Block-2
    """
    
    conv2_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv2_x1', kernel_initializer="glorot_normal")(conv1) 
    
    conv2_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv2_x2', kernel_initializer="glorot_normal")(conv2_x1)
    
    conv2_x3 =tf.keras.layers.BatchNormalization()(conv2_x2)
    
    conv2_x4 =tf.keras.layers.ReLU()(conv2_x3)   
    
    conv2_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv2_x5', kernel_initializer="glorot_normal")(conv2_x4)

    conv2_x6 =tf.keras.layers.BatchNormalization()(conv2_x5)      

    conv2_x7 =tf.keras.layers.ReLU()(conv2_x6)     

    conv2_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv2_x8', kernel_initializer="glorot_normal")(conv2_x7)
    
    conv2_x9 =tf.keras.layers.BatchNormalization()(conv2_x8)
    
    conv2_x10 =tf.keras.layers.ReLU()(conv2_x9)   
    
    conv2_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv2_x11', kernel_initializer="glorot_normal")(conv2_x10)        

    conv2_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv2_x12', kernel_initializer="glorot_normal")(conv2_x11)                                                                                                               
    
    conv2 = tf.keras.layers.Add(name='conv2')([conv2_x12, conv1])
    
    """
    Block-3
    """
    
    conv3_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv3_x1', kernel_initializer="glorot_normal")(conv2) 
    
    conv3_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv3_x2', kernel_initializer="glorot_normal")(conv3_x1)
    
    conv3_x3 =tf.keras.layers.BatchNormalization()(conv3_x2)
    
    conv3_x4 =tf.keras.layers.ReLU()(conv3_x3)   
    
    conv3_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv3_x5', kernel_initializer="glorot_normal")(conv3_x4)

    conv3_x6 =tf.keras.layers.BatchNormalization()(conv3_x5)      

    conv3_x7 =tf.keras.layers.ReLU()(conv3_x6)     

    conv3_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv3_x8', kernel_initializer="glorot_normal")(conv3_x7)
    
    conv3_x9 =tf.keras.layers.BatchNormalization()(conv3_x8)
    
    conv3_x10 =tf.keras.layers.ReLU()(conv3_x9)   
    
    conv3_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv3_x11', kernel_initializer="glorot_normal")(conv3_x10)        

    conv3_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv3_x12', kernel_initializer="glorot_normal")(conv3_x11)                                                                                                               
    
    conv3 = tf.keras.layers.Add(name='conv3')([conv3_x12, conv2])
    
    """
    Block-4
    """
    
    conv4_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv4_x1', kernel_initializer="glorot_normal")(conv3) 
    
    conv4_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv4_x2', kernel_initializer="glorot_normal")(conv4_x1)
    
    conv4_x3 =tf.keras.layers.BatchNormalization()(conv4_x2)
    
    conv4_x4 =tf.keras.layers.ReLU()(conv4_x3)   
    
    conv4_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv4_x5', kernel_initializer="glorot_normal")(conv4_x4)

    conv4_x6 =tf.keras.layers.BatchNormalization()(conv4_x5)      

    conv4_x7 =tf.keras.layers.ReLU()(conv4_x6)     

    conv4_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv4_x8', kernel_initializer="glorot_normal")(conv4_x7)
    
    conv4_x9 =tf.keras.layers.BatchNormalization()(conv4_x8)
    
    conv4_x10 =tf.keras.layers.ReLU()(conv4_x9)   
    
    conv4_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv4_x11', kernel_initializer="glorot_normal")(conv4_x10)        

    conv4_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv4_x12', kernel_initializer="glorot_normal")(conv4_x11)                                                                                                               
    
    conv4 = tf.keras.layers.Add(name='conv4')([conv4_x12, conv3])
    
    """
    Block-5
    """
    
    conv5_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv5_x1', kernel_initializer="glorot_normal")(conv4) 
    
    conv5_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv5_x2', kernel_initializer="glorot_normal")(conv5_x1)
    
    conv5_x3 =tf.keras.layers.BatchNormalization()(conv5_x2)
    
    conv5_x4 =tf.keras.layers.ReLU()(conv5_x3)   
    
    conv5_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv5_x5', kernel_initializer="glorot_normal")(conv5_x4)

    conv5_x6 =tf.keras.layers.BatchNormalization()(conv5_x5)      

    conv5_x7 =tf.keras.layers.ReLU()(conv5_x6)     

    conv5_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv5_x8', kernel_initializer="glorot_normal")(conv5_x7)
    
    conv5_x9 =tf.keras.layers.BatchNormalization()(conv5_x8)
    
    conv5_x10 =tf.keras.layers.ReLU()(conv5_x9)   
    
    conv5_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv5_x11', kernel_initializer="glorot_normal")(conv5_x10)        

    conv5_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv5_x12', kernel_initializer="glorot_normal")(conv5_x11)                                                                                                               
    
    conv5 = tf.keras.layers.Add(name='conv5')([conv5_x12, conv4])
    
    """
    Block-6
    """
    
    conv6_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv6_x1', kernel_initializer="glorot_normal")(conv5) 
    
    conv6_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv6_x2', kernel_initializer="glorot_normal")(conv6_x1)
    
    conv6_x3 =tf.keras.layers.BatchNormalization()(conv6_x2)
    
    conv6_x4 =tf.keras.layers.ReLU()(conv6_x3)   
    
    conv6_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv6_x5', kernel_initializer="glorot_normal")(conv6_x4)

    conv6_x6 =tf.keras.layers.BatchNormalization()(conv6_x5)      

    conv6_x7 =tf.keras.layers.ReLU()(conv6_x6)     

    conv6_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv6_x8', kernel_initializer="glorot_normal")(conv6_x7)
    
    conv6_x9 =tf.keras.layers.BatchNormalization()(conv6_x8)
    
    conv6_x10 =tf.keras.layers.ReLU()(conv6_x9)   
    
    conv6_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv6_x11', kernel_initializer="glorot_normal")(conv6_x10)        

    conv6_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv6_x12', kernel_initializer="glorot_normal")(conv6_x11)                                                                                                               
    
    conv6 = tf.keras.layers.Add(name='conv6')([conv6_x12, conv5])
    
    """
    Block-7
    """
    
    conv7_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv7_x1', kernel_initializer="glorot_normal")(conv6) 
    
    conv7_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv7_x2', kernel_initializer="glorot_normal")(conv7_x1)
    
    conv7_x3 =tf.keras.layers.BatchNormalization()(conv7_x2)
    
    conv7_x4 =tf.keras.layers.ReLU()(conv7_x3)   
    
    conv7_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv7_x5', kernel_initializer="glorot_normal")(conv7_x4)

    conv7_x6 =tf.keras.layers.BatchNormalization()(conv7_x5)      

    conv7_x7 =tf.keras.layers.ReLU()(conv7_x6)     

    conv7_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv7_x8', kernel_initializer="glorot_normal")(conv7_x7)
    
    conv7_x9 =tf.keras.layers.BatchNormalization()(conv7_x8)
    
    conv7_x10 =tf.keras.layers.ReLU()(conv7_x9)   
    
    conv7_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv7_x11', kernel_initializer="glorot_normal")(conv7_x10)        

    conv7_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv7_x12', kernel_initializer="glorot_normal")(conv7_x11)                                                                                                               
    
    conv7 = tf.keras.layers.Add(name='conv7')([conv7_x12, conv6])
    
    """
    Block-8
    """
    
    conv8_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv8_x1', kernel_initializer="glorot_normal")(conv7) 
    
    conv8_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv8_x2', kernel_initializer="glorot_normal")(conv8_x1)
    
    conv8_x3 =tf.keras.layers.BatchNormalization()(conv8_x2)
    
    conv8_x4 =tf.keras.layers.ReLU()(conv8_x3)   
    
    conv8_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv8_x5', kernel_initializer="glorot_normal")(conv8_x4)

    conv8_x6 =tf.keras.layers.BatchNormalization()(conv8_x5)      

    conv8_x7 =tf.keras.layers.ReLU()(conv8_x6)     

    conv8_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv8_x8', kernel_initializer="glorot_normal")(conv8_x7)
    
    conv8_x9 =tf.keras.layers.BatchNormalization()(conv8_x8)
    
    conv8_x10 =tf.keras.layers.ReLU()(conv8_x9)   
    
    conv8_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv8_x11', kernel_initializer="glorot_normal")(conv8_x10)        

    conv8_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv8_x12', kernel_initializer="glorot_normal")(conv8_x11)                                                                                                               
    
    conv8 = tf.keras.layers.Add(name='conv8')([conv8_x12, conv7])
    
    """
    Block-9
    """
    
    conv9_x1 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv9_x1', kernel_initializer="glorot_normal")(conv8) 
    
    conv9_x2 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv9_x2', kernel_initializer="glorot_normal")(conv9_x1)
    
    conv9_x3 =tf.keras.layers.BatchNormalization()(conv9_x2)
    
    conv9_x4 =tf.keras.layers.ReLU()(conv9_x3)   
    
    conv9_x5 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv9_x5', kernel_initializer="glorot_normal")(conv9_x4)

    conv9_x6 =tf.keras.layers.BatchNormalization()(conv9_x5)      

    conv9_x7 =tf.keras.layers.ReLU()(conv9_x6)  
    
    conv9_x8 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,   name='conv9_x8', kernel_initializer="glorot_normal")(conv9_x7)
    
    conv9_x9 =tf.keras.layers.BatchNormalization()(conv9_x8)
    
    conv9_x10 =tf.keras.layers.ReLU()(conv9_x9)   
    
    conv9_x11 = tf.keras.layers.Conv2D(32, [3,3], padding='SAME', use_bias=False,  name='conv9_x11', kernel_initializer="glorot_normal")(conv9_x10)        

    conv9_x12 = tf.keras.layers.Conv2D(1, [3,3], padding='SAME', use_bias=False,  name='conv9_x12', kernel_initializer="glorot_normal")(conv9_x11)                                                                                                               
    
    conv9 = tf.keras.layers.Add(name='conv9')([conv9_x12, conv8])
    
    output = tf.keras.layers.Reshape((1089,), name='output')(conv9)
    
    model = tf.keras.Model(inputs=[inp] , outputs=[output], name=name_model)
    
    
    
    model.summary()
    
    
    
    model.compile(optimizer=Adam(learning_rate=0.0001),\
                    loss="mse",\
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
    # plt.plot(history.history['val_loss'])
    # plt.title('AMP-Net Reconstruction Gauss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    # plt.plot(history.history['accuracy'])
    # # plt.plot(history.history['val_accuracy'])
    # plt.title('AMP-Net Reconstruction Gauss')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    
    
    return model, history, csv_logger
    
    
    