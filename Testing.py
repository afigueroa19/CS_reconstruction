# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:01:51 2022

@author: Alberto FIgueroa
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import time
from keras.layers import Input, Conv2D, Lambda, Reshape, Multiply, Add, Subtract, BatchNormalization
from keras.activations import relu
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from keras.callbacks import CSVLogger
from keras.utils import plot_model
import math



# x_test = np.load('x_test.npy' )
# x_train = np.load('x_train.npy')

# measure=50 # Las unicas opciones son 50 y 10
# if measure==10:
#     y_test = np.load('y_test_10.npy')    
#     y_train = np.load('y_train_10.npy')
    
# elif measure==50:
#     y_test = np.load('y_test_50.npy')    
#     y_train = np.load('y_train_50.npy')

def model(y_train, x_train, y_test, x_test):
   
    inp = tf.keras.Input(shape=(1089,))
    
    conv1_x2 = tf.keras.layers.Reshape((33, 33, 1), name='conv1_x2')(inp)
    
    conv1_x3 = tf.keras.layers.Conv2D(64, [11,11], padding='SAME', use_bias=False, name='conv1_x3', kernel_initializer="glorot_normal")(conv1_x2)
    
    conv1_x3 =tf.keras.layers.BatchNormalization()(conv1_x3)
    
    conv1_sl1 = tf.keras.layers.Conv2D(32, [1,1], padding='SAME', use_bias=False, activation='relu', name='conv1_sl1', kernel_initializer="glorot_normal") #Paso intermedio
    
    conv1_x4 = conv1_sl1(conv1_x3)
    
    conv1_x4 =tf.keras.layers.BatchNormalization()(conv1_x4)
    
    conv1_sl2 = tf.keras.layers.Conv2D(1, [7,7], padding='SAME', use_bias=False, name='conv1_sl2', kernel_initializer="glorot_normal")
    
    conv1_x5 = conv1_sl2(conv1_x4)
    
    conv1_x6 = tf.keras.layers.Add(name='conv1_x6')([conv1_x5, conv1_x2])
    

    
    
    model.summary()
    
    
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, to_file='model_plot.png')  
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=13, min_delta=0.00001)
    
    csv_logger = CSVLogger('DR2_recon_300epochs.log', separator=',', append=False)
    
    history = model.fit(y_train, x_train, epochs=300,validation_data=(y_test, x_test),
                    batch_size = 64, shuffle=True, callbacks=[csv_logger])
    
    # history = model.fit(y_train, x_train, epochs=50, validation_data=(y_test, x_test),
    #                 batch_size = 64, shuffle=True, callbacks=[callback, csv_logger])
    
    
    
    model.save('HSCNN-R_300epochs.h5')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('DR2_recon')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('DR2_recon')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    return model, csv_logger

# model(y_train,x_train, y_test, x_test)

def retraining (y_train, x_train):

    loaded_model = models.load_model('ISTAxRECON-BlockbyBlockV0.h5')
    
    # loaded_model.summary()
    
    csv_logger = CSVLogger('ISTAxRECON-BlockbyBlockV2.log', separator=',', append=False)
    
    history = loaded_model.fit(y_train, x_train, epochs=100, 
                batch_size = 64, shuffle=True, callbacks=[csv_logger])
    
    # loaded_model.save('ISTAxRECON-BlockbyBlockV3.h5')
    
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('50% ISTA Reconstruction')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('50% ISTA Reconstruction')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    
    return loaded_model, history
# model, history = retraining (y_train, x_train)
# model.save('ISTAxRECON-BlockbyBlockV3.h5')

def quality(y_train):
    
    model = models.load_model("Initial_ReconV2_88912_Gauss50_V1.h5")
    
    
    prediction = model.predict(y_train)
    
    # length = int(np.size(y_test)/(33*33))
                 
    #   plt.subplot(1, 3, 1)
    # plt.imshow(np.reshape(y_test[0],(33,33)), cmap="gray", vmin=0,vmax=1)
    # plt.title('Pseudo-Inversa', fontsize=10 )
    # plt.subplot(1, 3, 2)
    # plt.imshow(x_test[0], cmap="gray", vmin=0, vmax=1)
    # plt.title('Original', fontsize=10 )
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.reshape(prediction[0],(33,33)), cmap="gray", vmin=0, vmax=1)
    # plt.title('Reconstruccion Inicial', fontsize=10 )
    # plt.show()
    # PSNR_value=0
    # SSIM_value=0
    
    # for i in range(length):
    #     PSNR_value = PSNR_value + PSNR(x_test[i], np.reshape(y_test[i],(33,33)))
    #     SSIM_value = SSIM_value + ssim(x_test[i], np.reshape(y_test[i],(33,33)),
    #                         data_range=np.reshape(y_test[i],(33,33)).max() - np.reshape(y_test[i],(33,33)).min())
    # PSNR_value = PSNR_value/length
    # SSIM_value = SSIM_value/length
    
    # print(f"Pseudo_reconstruction- PSNR_value is {PSNR_value} dB, SSIM_value is {SSIM_value}")
    
    # PSNR_value=0
    # SSIM_value=0
    
    # for i in range(length):
    #     PSNR_value = PSNR_value + PSNR(x_test[i], np.reshape(prediction[i],(33,33)))
    #     SSIM_value = SSIM_value + ssim(x_test[i], np.reshape(prediction[i],(33,33)),
    #                         data_range=np.reshape(prediction[i],(33,33)).max() - np.reshape(prediction[i],(33,33)).min())
    # PSNR_value = PSNR_value/length
    # SSIM_value = SSIM_value/length
    
    # print(f"CNN_reconstruction- PSNR_value is {PSNR_value} dB, SSIM_value is {SSIM_value}")
    
    return prediction
# prediction = quality(y_test, x_test)

def PSNR(original, compressed):
    original.astype(np.float32)
    compressed.astype(np.float32)
    mse = np.mean((original - compressed) ** 2)
    # rmse = np.sqrt(np.mean((original - compressed) ** 2))
    # N_rmse = np.sqrt(np.mean((original - compressed) ** 2))/255.0
    # print(mse)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


# def psnr(img1, img2):
#     img1.astype(np.float32)
#     img2.astype(np.float32)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def complete_reconstruction():
   
    Initial_Reconstruction=models.load_model("Initial_ReconV2_88912_Gauss01_V10.h5")
    Initial_Reconstruction.trainable=False

    Blocks=models.load_model('ISTAxRECON_88912_Gauss01_V1.h5')
    Blocks.trainable=True
    
    name_model = 'ISTAxRECON-Full-Gauss01.h5'
    # Block2=models.load_model('ISTAxRECON-2BlockV2.h5')
    # Block2.trainable=True
    
    # Block3=models.load_model('ISTAxRECON-3BlockV1.h5')
    # Block3.trainable=True
    
    # Block4=models.load_model('ISTAxRECON-4BlockV2.h5')
    # Block4.trainable=True
    
    # Block5=models.load_model('ISTAxRECON-5BlockV0.h5')
    # Block5.trainable=True
    
    # Block6=models.load_model('ISTAxRECON-6BlockV0.h5')
    # Block6.trainable=True
    
    # Block7=models.load_model('ISTAxRECON-7BlockV0.h5')
    # Block7.trainable=True
    

    model_2=tf.keras.Sequential([Initial_Reconstruction, Blocks],  name=name_model)
    # model_2.name="model_2"
    model_2.summary()
    
    model_2.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
    model_2.save(name_model)
    
    
    return 


    
    # aprom=0
    # bprom=0
    # for i in range (11):     
    #     im0, tiles, y_im_crop, input_im_crop=image_to_tiles(i,tile_size = 33)
    #     prediction = quality(y_im_crop, x_test)
    #     pasting_im_remake=pasting(i, prediction, tile_size=33)
    #     a=PSNR(im0, pasting_im_remake)
    #     aprom=aprom+a
    #     b=ssim(im0, pasting_im_remake,data_range=pasting_im_remake.max() - pasting_im_remake.min())
    #     bprom=bprom+b
    #     # print(a)
    #     # print(b)
    # aprom=aprom/11
    # bprom=bprom/11
    # print(aprom)
    # print(bprom)
    
    
    
    
    # aprom=0
    # bprom=0
    # for i in range (11):     
    #     im0, tiles, y_im_crop, input_im_crop=image_to_tiles(i,CS, tile_size = 33)
    #     model = models.load_model('Initial_Recon_88912_Gauss50_V4.h5')
    #     prediction = model.predict(y_im_crop)
    #     model = models.load_model('ISTAxRECON_88912_Gauss50_V0.h5')
    #     prediction = model.predict(prediction)
    #     pasting_im_remake, im0=pasting(i, prediction, im0, tile_size=33)
    #     a=PSNR(im0, pasting_im_remake)
    #     aprom=aprom+a
    #     b=ssim(im0, pasting_im_remake,data_range=pasting_im_remake.max() - pasting_im_remake.min())
    #     bprom=bprom+b
    # aprom=aprom/11
    # bprom=bprom/11
    # print(aprom)
    # print(bprom)
    
    











    
    
    
    
    
    
    
    
    