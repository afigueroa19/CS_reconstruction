# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:07:30 2022

@author: Alberto Figueroa

ISTAxRECON
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



def ISTAxRECON(y_train, x_train, i):
    name_model = f'V2_Gauss_Block{i}.h5'
    csv_name= f'V2_Gauss_Block{i}.log'

    
    '''
    ISTAxRECON block #1
    
    '''
    
    
    inp = tf.keras.Input(shape=(1089,))

    conv1_x1 = tf.keras.layers.Reshape((33, 33, 1), name='conv1_x1')(inp) ###Reshape al inicio
    
    conv1_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, name='conv1_x3', kernel_initializer="glorot_normal")(conv1_x1) ###Primera capa
    
    conv1_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv1_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv1_x4 = conv1_sl1(conv1_x3)###Segunda capa
    
    conv1_sl0 =tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv1_sl0', kernel_initializer="glorot_normal")
    
    conv1_x2 = conv1_sl0(conv1_x4)###Tercera capa
    
    conv1_x5 = Multiply(name='conv1_x5')([Lambda(lambda x: K.sign(x))(conv1_x2), Lambda(lambda x: relu(x - 0.1))(Lambda(lambda x: K.abs(x))(conv1_x2))])####Cuarta capa
    
    conv1_sl3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv1_sl3', kernel_initializer="glorot_normal")
    
    conv1_x6 = conv1_sl3(conv1_x5)###Quinta capa
    
    conv1_sl4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False,  name='conv1_sl4', kernel_initializer="glorot_normal")
    
    conv1_x66 = conv1_sl4(conv1_x6)####Sexta capa
    
    conv1_sl5 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv1_sl5', kernel_initializer="glorot_normal")
    
    conv1_x7 = conv1_sl5(conv1_x66) ####Septima capa
    
    conv1 = tf.keras.layers.Add(name='conv1')([conv1_x7, conv1_x1]) ####Octaba capa/ Fin del primer bloque
    
    
    
    
    '''
      ISTAxRECON block #2
    
    '''

    
    conv2_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv2_x3', kernel_initializer="glorot_normal")(conv1)
    
    conv2_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv2_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv2_x4 = conv2_sl1(conv2_x3)
    
    conv2_sl2 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv2_sl2', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv2_x44 = conv2_sl2(conv2_x4)
    
    conv2_x5 = Multiply(name='conv2_x5')([Lambda(lambda x: K.sign(x))(conv2_x44), Lambda(lambda x: relu(x - 0.1))(Lambda(lambda x: K.abs(x))(conv2_x44))])
    
    conv2_sl3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv2_sl3', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv2_x6 = conv2_sl3(conv2_x5)
    
    conv2_sl4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv2_sl4', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv2_x66 = conv2_sl4(conv2_x6)
    
    conv2_sl5=tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv2_sl5', kernel_initializer="glorot_normal")
    
    conv2_x7 = conv2_sl5(conv2_x66)
    
    conv2 = tf.keras.layers.Add(name='conv2')([conv2_x7, conv1])
    
    
    '''
      ISTAxRECON block #3
    
    '''
    
    conv3_x2 = tf.keras.layers.Reshape((33, 33, 1), name='conv3_x2')(conv2)
    
    conv3_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv3_x3', kernel_initializer="glorot_normal")(conv3_x2)
    
    conv3_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv3_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv3_x4 = conv3_sl1(conv3_x3)
    
    conv3_sl2 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv3_sl2', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv3_x44 = conv3_sl2(conv3_x4)
    
    conv3_x5 = Multiply(name='conv3_x5')([Lambda(lambda x: K.sign(x))(conv3_x44), Lambda(lambda x: relu(x - 0.1))(Lambda(lambda x: K.abs(x))(conv3_x44))])
    
    conv3_sl3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv3_sl3', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv3_x6 = conv3_sl3(conv3_x5)
    
    conv3_sl4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv3_sl4', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv3_x66 = conv3_sl4(conv3_x6)
    
    conv3_sl5=tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv3_sl5', kernel_initializer="glorot_normal")
    
    conv3_x7 = conv3_sl5(conv3_x66)
    
    conv3 = tf.keras.layers.Add(name='conv3')([conv3_x7, conv2])

    
    '''
      ISTAxRECON block #4
    
    '''
  
    
    conv4_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv4_x3', kernel_initializer="glorot_normal")(conv3)
    
    conv4_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv4_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv4_x4 = conv4_sl1(conv4_x3)
    
    conv4_sl2 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv4_sl2', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv4_x44 = conv4_sl2(conv4_x4)
    
    conv4_x5 = Multiply(name='conv4_x5')([Lambda(lambda x: K.sign(x))(conv4_x44), Lambda(lambda x: relu(x - 0.1))(Lambda(lambda x: K.abs(x))(conv4_x44))])
    
    conv4_sl3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv4_sl3', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv4_x6 = conv4_sl3(conv4_x5)
    
    conv4_sl4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv4_sl4', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv4_x66 = conv4_sl4(conv4_x6)
    
    conv4_sl5=tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv4_sl5', kernel_initializer="glorot_normal")
    
    conv4_x7 = conv4_sl5(conv4_x66)
    
    conv4 = tf.keras.layers.Add(name='conv4')([conv4_x7, conv3])

    
    '''
      ISTAxRECON block #5
    
    '''
    

    conv5_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv5_x3', kernel_initializer="glorot_normal")(conv4)
    
    conv5_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv5_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv5_x4 = conv5_sl1(conv5_x3)
    
    conv5_sl2 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv5_sl2', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv5_x44 = conv5_sl2(conv5_x4)
    
    conv5_x5 = Multiply(name='conv5_x5')([Lambda(lambda x: K.sign(x))(conv5_x44), Lambda(lambda x: relu(x - 0.1))(Lambda(lambda x: K.abs(x))(conv5_x44))])
    
    conv5_sl3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv5_sl3', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv5_x6 = conv5_sl3(conv5_x5)
    
    conv5_sl4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv5_sl4', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv5_x66 = conv5_sl4(conv5_x6)
    
    conv5_sl5=tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv5_sl5', kernel_initializer="glorot_normal")
    
    conv5_x7 = conv5_sl5(conv5_x66)
    
    conv5 = tf.keras.layers.Add(name='conv5')([conv5_x7, conv4])
    
    
    
    
    
    '''
      ISTAxRECON block #6
    
    '''
    

    conv6_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv6_x3', kernel_initializer="glorot_normal")(conv5)
    
    conv6_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv6_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv6_x4 = conv6_sl1(conv6_x3)
    
    conv6_sl2 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv6_sl2', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv6_x44 = conv6_sl2(conv6_x4)
    
    conv6_x5 = Multiply(name='conv6_x5')([Lambda(lambda x: K.sign(x))(conv6_x44), Lambda(lambda x: relu(x - 0.1))(Lambda(lambda x: K.abs(x))(conv6_x44))])
    
    conv6_sl3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv6_sl3', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv6_x6 = conv6_sl3(conv6_x5)
    
    conv6_sl4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv6_sl4', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv6_x66 = conv6_sl4(conv6_x6)
    
    conv6_sl5=tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv6_sl5', kernel_initializer="glorot_normal")
    
    conv6_x7 = conv6_sl5(conv6_x66)
    
    conv6 = tf.keras.layers.Add(name='conv6_x8')([conv6_x7, conv5])
    
    
    
    '''
      ISTAxRECON block #7
    
    '''
    

    conv7_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv7_x3', kernel_initializer="glorot_normal")(conv6)
    
    conv7_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv7_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv7_x4 = conv7_sl1(conv7_x3)
    
    conv7_sl2 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv7_sl2', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv7_x44 = conv7_sl2(conv7_x4)
    
    conv7_x5 = Multiply(name='conv7_x5')([Lambda(lambda x: K.sign(x))(conv7_x44), Lambda(lambda x: relu(x - 0.1))(Lambda(lambda x: K.abs(x))(conv7_x44))])
    
    conv7_sl3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, activation='relu', name='conv7_sl3', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv7_x6 = conv7_sl3(conv7_x5)
    
    conv7_sl4 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv7_sl4', kernel_initializer="glorot_normal")#Paso intermedio
    
    conv7_x66 = conv7_sl4(conv7_x6)
    
    conv7_sl5=tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv7_sl5', kernel_initializer="glorot_normal")
    
    conv7_x7 = conv7_sl5(conv7_x66)
    
    conv7_x8 = tf.keras.layers.Add(name='conv7_x8')([conv7_x7, conv6])
    
    output = Reshape((1089,), name='output')(conv7_x8)
    
    
    """
    
    Training
    
    """
    
    model = tf.keras.Model(inputs=[inp] , outputs=[output], name=name_model)
    
    model.summary()
    
    model.compile(optimizer= Adam(learning_rate=0.0001),\
                  metrics=['accuracy'],  loss="mse")
    
    path_CSV = f'./IstaPrueba/{csv_name}'

    csv_logger = CSVLogger(path_CSV , separator=',', append=False) 

    history = model.fit(x = y_train, y = x_train, epochs=200,\
                          batch_size = 64, shuffle=True, callbacks=[csv_logger])
  
    path_model = f'./IstaPrueba/{name_model}'
    
    model.save(path_model) 
    
    
    # plt.plot(history.history['loss'])
    # plt.title('ISTAxRECON Reconstruction Gauss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    
    # plt.plot(history.history['accuracy'])
    # plt.title('ISTAxRECON Reconstruction Gauss')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    
    
    
    return model, history, csv_logger
