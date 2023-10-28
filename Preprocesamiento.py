# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:47:54 2022

@author: Alberto Figueroa
"""
import scipy.io as sio
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import cv2
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

train = sio.loadmat('Training_Data_Img91.mat')

inputs=train['labels']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

measure=50



def Mask ():###Para mascara Gaussiana, esto se puede optimixzar un monton
    a=[]
    for i in range(545):
        a.append([])
    
    for i in range(545):
        
        k = 1089
        a[i]=np.random.normal(loc=0,scale=1, size=k)
    a=np.transpose(np.array(a))
    
    
    CS=gram_schmidt(a)
    
    CS=np.transpose(CS)
    
    np.save("CS_Matrix_1089_50_Gauss",CS)
    
    return CS

def gram_schmidt(A):
    
    (n, m) = A.shape
    
    for i in range(m):
        
        q = A[:, i] # i-th column of A
        
        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]
        
        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")
        
        # normalize q
        q = q / np.sqrt(np.dot(q, q))
        
        # write the vector back in the matrix
        A[:, i] = q
    return A


def TrTest (measure, train_images, test_images, CS):

    y_train=[]
    x_train=[]
    y_test=[]
    x_test=[]

    for i in range(49999):

        y_train.append([])
        x_train.append([])
        
    for i in range(9999):

        y_test.append([])
        x_test.append([])
        
    for i in range(49999):
        x=Image.fromarray(train_images[i])
        x=ImageOps.grayscale(x)
        x=x.resize((33,33))
        x=np.array(x)
        x=x/255
        x=np.reshape(x,(1089))
        x_train[i]= x
        # x=np.reshape(x,(1089))
        y_train[i]=np.dot(CS,x)
    
    for i in range(9999):
        x=Image.fromarray(test_images[i])
        x=ImageOps.grayscale(x)
        x=x.resize((33,33))
        x=np.array(x)
        x=x/255
        x=np.reshape(x,(1089))
        x_test[i]= x
        # x=np.reshape(x,(1089))
        y_test[i]=np.dot(CS,x)
        
    x_train=np.array(x_train)
    
    x_test=np.array(x_test)
    
    y_train=np.array(y_train)
    
    y_test=np.array(y_test)
    print("Termino el preprocesamiento de datos1")
    
    return y_train, x_train, y_test, x_test

# y_train1, x_train1, y_test1, x_test1 = (TrTest (measure, train_images, test_images, CS))
# y_train, x_train, y_test, x_test = (TrTest2 (inputs, CS))
# y_train, x_train, y_test, x_test = concat(y_train1, x_train1, y_test1, x_test1, y_train2, x_train2, y_test2, x_test2)
# del y_train1, x_train1, y_test1, x_test1, y_train2, x_train2, y_test2, x_test2
# del inputs, measure, test_images, test_labels, train, train_images,train_labels,x_test,x_train
# x_test = np.load('x_test.npy' )
# x_train = np.load('x_train.npy')
# y_test, y_train = Input_creator( y_test, y_train)
# np.save("y_train_50_Gauss",y_train)
# np.save("y_test_50_Gauss",y_test)
train = sio.loadmat('Training_Data_Img91.mat')

inputs=train['labels']

def TrTest2 (inputs, CS):
    
    CS=CS
    
    y_train=[]
    x_train=[]

    for i in range(88912):
        y_train.append([])
        x_train.append([])
    
    
    for i in range(88912):
        x=(inputs[i])
        x_train[i]= np.reshape(x,(1089,))
        y_train[i]=np.dot(CS,x)
 
    y_train=np.array(y_train)
    x_train=np.array(x_train)
    
    # [a1,b1,c1,d1,e1,f1,g1,h1,i1] = np.array_split(x_train, 9)
    
    # x_train2 = np.concatenate((a1,b1,c1,d1,e1,f1,g1,h1), axis=0)
    
    # x_test2= i1
    
    # [a2,b2,c2,d2,e2,f2,g2,h2,i2] = np.array_split(y_train, 9)
    
    # y_train2 = np.concatenate((a2,b2,c2,d2,e2,f2,g2,h2), axis=0) 
    
    
    # y_test2= i2
    
    # y_train2=np.array(y_train2)
    # x_train2=np.array(x_train2)
    
    # y_test2=np.array(y_test2)
    # x_test2=np.array(x_test2)
    print("Termino el preprocesamiento de datos2")
    return y_train, x_train
    
# y_train2, x_train2, y_test2, x_test2 = (TrTest2 (inputs, CS))


def concat(y_train1, x_train1, y_test1, x_test1, y_train2, x_train2, y_test2, x_test2):
    
    y_train=[]
    x_train=[]
    y_test=[]
    x_test=[]
    
    y_train= np.concatenate((y_train1, y_train2), axis=0)
    y_test= np.concatenate((y_test1, y_test2), axis=0)
    
    x_train= np.concatenate((x_train1, x_train2), axis=0)
    x_test= np.concatenate((x_test1, x_test2), axis=0)
    
    del y_train1, x_train1, y_test1, x_test1, y_train2, x_train2, y_test2, x_test2
    
    return y_train, x_train, y_test, x_test
    
# y_train, x_train, y_test, x_test = concat(y_train1, x_train1, y_test1, x_test1, y_train2, x_train2, y_test2, x_test2)

def Input_creator(y_train, CS, x_train,PhiInv_input):
    
    # Phi_input=CS.transpose()
    # XX = x_train.transpose() #Igual a x_train.transpose()
    # BB = y_train #Igual a y_train
    # BBB = np.dot(BB, BB.transpose())
    # CCC = np.dot(XX, BB.transpose())
    # PhiT_ = np.dot(CCC, np.linalg.inv(BBB))
    # del XX, BB, BBB
    # PhiInv_input = PhiT_.transpose()
    
    # pseudo_inv=np.linalg.pinv(CS, rcond=1e-15, hermitian=False)
    pseudo_inv=PhiInv_input
    y_train2=[]
    
    for i in range(88912):
       y_train2.append([])
       
    for i in range(88912):
        y_train2[i]=np.dot(y_train[i], pseudo_inv)
         
    y_train2=np.array(y_train2)
    
    print("Termino la aplicacion de la pseudo inversa")
    
    return  y_train2 
    
    
    #  y_test, y_train = Input_creator( y_test, y_train)
    
def image_to_tiles(i, CS, tile_size = 33):
   
    if i==0:
        im0 = Image.open("monarch.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("monarch.tif")
    elif i==1:
        im0 = Image.open("fingerprint.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,16,0,16,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("fingerprint.tif")
    elif i==2:
        im0 = Image.open("flinstones.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,16,0,16,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("flinstones.tif")
    elif i==3:
        im0 = Image.open("lena256.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("lena256.tif")
    elif i==4:
        im0 = Image.open("parrots.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("parrots.tif")
    elif i==5:
        im0 = Image.open("barbara.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("barbara.tif")
    elif i==6:
        im0 = Image.open("boats.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("boats.tif")
    elif i==7:
        im0 = Image.open("cameraman.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("cameraman.tif")
    elif i==8:
        im0 = Image.open("foreman.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("foreman.tif")
    elif i==9:
        im0 = Image.open("house.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("house.tif")
    elif i==10:
        im0 = Image.open("peppers256.tif")
        im0=np.array(im0)/255
        im0 = cv2.copyMakeBorder(im0,0,8,0,8,cv2.BORDER_CONSTANT,value=1)
        im0 = Image.fromarray(im0)
        print("peppers256.tif")
    else:
        print("Error seleccionando imagen")
        
        
    w,h=np.shape(im0)
    
    row_count = int(w/tile_size)
    col_count = int(h/tile_size)
    
    n_slices = int(row_count*col_count)
    
    # Image info
    
    # print(f'Dimensions: w:{w} h:{h}')
    # print(f'Tile count: {n_slices}')

    CS= CS
    pseudo_inv=np.linalg.pinv(CS, rcond=1e-15, hermitian=False)
    tiles = []
    y_im_crop=[]
    input_im_crop=[]
    
    
    
    for i in range(n_slices):
        tiles.append([])
        y_im_crop.append([])
        input_im_crop.append([])
        
    for row in range(row_count):
        c=row*tile_size
        d=(row+1)*tile_size
        for column in range(col_count):
            a=0+tile_size*column
            b=tile_size+tile_size*column
            tiles[column+row_count*row]=np.array(im0.crop((a,c,b,d)))
           
    # tiles=np.array(tiles)/255
    
    for i in range(n_slices):
        y_im_crop[i]=np.dot(CS,np.reshape(tiles[i],(1089)))



    y_im_crop=np.array(y_im_crop)
    
    for i in range(n_slices):
        # input_im_crop[i]=np.dot(pseudo_inv, y_im_crop[i])
        input_im_crop[i]=np.reshape(np.dot(pseudo_inv, y_im_crop[i]),(33,33))#Para que tenga forma de cuadrados
    
    
    input_im_crop=np.array(input_im_crop)
    im0=np.array(im0)
    
    return im0, tiles, y_im_crop, input_im_crop
    

def pasting(i, prediction, im0, tile_size=33):
    
        
    w,h=np.shape(im0)
    
    row_count = int(w/tile_size)
    col_count = int(h/tile_size)
    
    n_slices = int(row_count*col_count)
    
    pasting=[]
    
    prediction2=[]
    
    for h in range(64):
        prediction2.append([])
        
    for h in range(64):
        prediction2[h]=np.reshape(prediction[h],(33,33))
    
    prediction2=np.array(prediction2)
    prediction=prediction2
    
    for a in range(row_count):
        pasting.append([])
    
        
    for row in range(row_count):
        
        for column in range(col_count):
            if column==0:
                pasting[row]=prediction[column+row_count*row]
            else:
                pasting[row]=np.concatenate((pasting[row],prediction[column+row_count*row]), axis=1)

    
    for row in range(row_count):
        if row==0:
            pasting_im_remake=pasting[row]
        else:
            pasting_im_remake = np.concatenate(( pasting_im_remake, pasting[row]), axis=0)
            
    # scaler = MinMaxScaler()
    
    # pasting_im_remake = scaler.fit_transform(pasting_im_remake)
    # pasting_im_remake=np.array(pasting_im_remake)
    
    # for row in range(w):
        
    #     for column in range(h):
    #         if pasting_im_remake[w-1,h-1] < 0:
    #             pasting_im_remake[w-1,h-1] = 0
                
    #         elif pasting_im_remake[w-1,h-1] > 1:
    #             pasting_im_remake[w-1,h-1] = 1
    
    # im0 = Image.fromarray(im0/255)
    im0 = Image.fromarray(im0)
    pasting_im_remake = Image.fromarray(pasting_im_remake)
    
    if i==0:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==1:
        im0=np.array(im0.crop((0,0,512,512)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,512,512)))
        
    elif i==2:
        im0=np.array(im0.crop((0,0,512,512)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,512,512)))
        
    elif i==3:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==4:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==5:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==6:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==7:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==8:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==9:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    elif i==10:
        im0=np.array(im0.crop((0,0,256,256)))
        pasting_im_remake=np.array(pasting_im_remake.crop((0,0,256,256)))
        
    else:
        print("Error cropeando")
    
    
    plt.imshow(pasting_im_remake, cmap="gray", vmin=0,vmax=1)
    plt.axis('off')
    plt.grid(b=None)
    plt.show()
    
    pasting_im_remake=np.array(pasting_im_remake)
    im0=np.array(im0)
    
    
    return  pasting_im_remake, im0
# aprom=0
# bprom=0
# for i in range (11):     
        
#     Iorg, row, col, Ipad, row_new, col_new=imread_CS_py(i)
#     y_im_crop=img2col_py(Ipad, block_size,CS)
#     prediction = quality(y_im_crop, x_test)
#     X_rec=col2im_CS_py(prediction, row, col, row_new, col_new)
#     a=PSNR(im0, pasting_im_remake)
#     aprom=aprom+a
#     b=ssim(im0, pasting_im_remake,data_range=pasting_im_remake.max() - pasting_im_remake.min())
#     bprom=bprom+b
# aprom=aprom/11
# bprom=bprom/11
# print(aprom)
# print(bprom)
def imread_CS_py(i):
    if i==0:
        im0 = Image.open("monarch.tif")
        print("monarch.tif")
    elif i==1:
        im0 = Image.open("fingerprint.tif")
        print("fingerprint.tif")
    elif i==2:
        im0 = Image.open("flinstones.tif")
        print("flinstones.tif")
    elif i==3:
        im0 = Image.open("lena256.tif")
        print("lena256.tif")
    elif i==4:
        im0 = Image.open("parrots.tif")
        print("parrots.tif")
    elif i==5:
        im0 = Image.open("barbara.tif")
        print("barbara.tif")
    elif i==6:
        im0 = Image.open("boats.tif")
        print("boats.tif")
    elif i==7:
        im0 = Image.open("cameraman.tif")
        print("cameraman.tif")
    elif i==8:
        im0 = Image.open("foreman.tif")
        print("foreman.tif")
    elif i==9:
        im0 = Image.open("house.tif")
        print("house.tif")
    elif i==10:
        im0 = Image.open("peppers256.tif")
        print("peppers256.tif")
    else:
        print("Error seleccionando imagen")
    # im0 = cube[:,:,i]
    block_size = 33
    Iorg = np.array(im0, dtype='float64')#ejemplo: monarca
    [row, col] = Iorg.shape #[256,256]
    row_pad = block_size - np.mod(row, block_size) #33-25=8
    col_pad = block_size - np.mod(col, block_size) #33-25=8
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1) #cada fila quedara con 264 elementos
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0) #Se agregan la cantidad de filas calculadas en row_pad
    [row_new, col_new] = Ipad.shape
    return Iorg, row, col, Ipad, row_new, col_new, block_size

def img2col_py(Ipad, block_size,CS, pseudo_inv):
    [row, col] = Ipad.shape # Continuacion ejemplo monarca--dimensiones[264,264]
    row_block = row / block_size #8
    col_block = col / block_size #8
    block_num = int(row_block * col_block) #64
    img_col = np.zeros([block_size ** 2, block_num])
    count = 0
    
    #parte la imagen en bloques pero lo mantiene como vector
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    img_col=img_col.transpose()/255
    #Seccion mía
    y_im_crop=[]
    for i in range(block_num):
        y_im_crop.append([])
    for i in range(block_num):
        y_im_crop[i]=np.dot(CS,img_col[i])



    y_im_crop=np.array(y_im_crop)
    
    
    
    # pseudo_inv=np.linalg.pinv(CS, rcond=1e-15, hermitian=False)
    # pseudo_inv=pseudo_inv.transpose()
    
    y_train2=[]
    
    for i in range(block_num):
       y_train2.append([])
       
    for i in range(block_num):
        y_train2[i]=np.dot(y_im_crop[i], pseudo_inv)
         
       
    
    
    y_train2=np.array(y_train2)
    
    return y_im_crop, block_num, y_train2

def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
            count = count + 1
    X_rec = X0_rec[:row, :col]
    X_rec=X_rec*255
    
    plt.imshow(X_rec,  vmin=0,vmax=255, cmap='gray')
    plt.axis('off')
    plt.grid(b=None)
    plt.show()
    return X_rec
  
"""

Para Gaussiana


  
CS = np.load('CS_Gauss_01.npy')
aprom=0
bprom=0    
for i in range(11):
    Iorg, row, col, Ipad, row_new, col_new, block_size=imread_CS_py(i)
    y_im_crop, block_num, y_train2=img2col_py(Ipad, block_size,CS)
    model = models.load_model('Initial_ReconV2_88912_Gauss01_V10.h5')
    prediction = model.predict(y_im_crop)
    model = models.load_model('ISTAxRECON_88912_Gauss01_V0.h5')
    prediction = model.predict(prediction)
    X_rec=col2im_CS_py(prediction.transpose(), row, col, row_new, col_new)
    a=PSNR(Iorg, X_rec)
    aprom=aprom+a
    b=ssim(Iorg, X_rec,data_range=X_rec.max() - X_rec.min())
    bprom=bprom+b
    # print(a,";" ,b ) 

aprom=aprom/11
bprom=bprom/11
print(aprom,";",bprom) 


"""

"""

Para Binaria


CS = np.load('CS_Binary_50.npy')
pseudo_inv=np.linalg.pinv(CS, rcond=1e-15, hermitian=False)
aprom=0
bprom=0    
for i in range(11):
    Iorg, row, col, Ipad, row_new, col_new, block_size=imread_CS_py(i)
    y_im_crop, block_num, y_train2=img2col_py(Ipad, block_size,CS)
    model = models.load_model('RECON_NET_88912_Binary50_V0.h5')
    prediction = model.predict(y_train2)
    X_rec=col2im_CS_py(prediction.transpose(), row, col, row_new, col_new)
    a=PSNR(Iorg, X_rec)
    aprom=aprom+a
    b=ssim(Iorg, X_rec,data_range=X_rec.max() - X_rec.min())
    bprom=bprom+b
    # print(a,";" ,b ) 

aprom=aprom/11
bprom=bprom/11
print(aprom,";", bprom) 

"""

"""
Guardado de imagenes sin marco Gauss

CS = np.load('CS_Gauss_50.npy')
aprom=0
bprom=0    
Iorg, row, col, Ipad, row_new, col_new, block_size=imread_CS_py(0)
y_im_crop, block_num, y_train2=img2col_py(Ipad, block_size,CS)
model = models.load_model('Initial_ReconV2_88912_Gauss50_V1.h5')
prediction = model.predict(y_im_crop)
model = models.load_model('HSCNN-R_88912_Gauss50_V0.h5')
prediction = model.predict(prediction)
X_rec=col2im_CS_py(prediction.transpose(), row, col, row_new, col_new)
im0 = X_rec

im0=np.array(im0)


fig, ax = plt.subplots()

# prepare the demo image


extent = [0, 255, 0, 255]
Z2 = np.zeros([256, 256], dtype="d")
ny, nx = im0.shape
Z2[0:0+ny, 0:0+nx] = im0

# extent = [-3, 4, -4, 3]
ax.imshow(im0, extent=extent, interpolation="nearest",
          origin="upper", cmap="gray", vmin=0,vmax=255)
plt.axis('off')
plt.grid(b=None)
axins = zoomed_inset_axes(ax, 1.3, loc=3) # zoom = 6
axins.imshow(Z2, extent=extent, interpolation="nearest",
              origin="upper",  cmap="gray", vmin=0,vmax=255)

# sub region of the original image
x1, x2, y1, y2 =131, 197, 125, 191
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)
# plt.imshow(X_rec, cmap="gray", vmin=0,vmax=255)
plt.axis('off')
plt.grid(b=None)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.draw()
plt.savefig("parrot.png", bbox_inches='tight', pad_inches=0)
plt.show()


Iorg, row, col, Ipad, row_new, col_new, block_size=imread_CS_py(4)
y_im_crop, block_num, y_train2=img2col_py(Ipad, block_size,CS)
model = models.load_model('Initial_ReconV2_88912_Gauss50_V1.h5')
prediction = model.predict(y_im_crop)
model = models.load_model('HSCNN-R_88912_Gauss50_V0.h5')
prediction = model.predict(prediction)
X_rec=col2im_CS_py(prediction.transpose(), row, col, row_new, col_new)
im0 = X_rec

im0=np.array(im0)


fig, ax = plt.subplots()

# prepare the demo image


extent = [0, 255, 0, 255]
Z2 = np.zeros([256, 256], dtype="d")
ny, nx = im0.shape
Z2[0:0+ny, 0:0+nx] = im0

# extent = [-3, 4, -4, 3]
ax.imshow(im0, extent=extent, interpolation="nearest",
          origin="upper", cmap="gray", vmin=0,vmax=255)
plt.axis('off')
plt.grid(b=None)
axins = zoomed_inset_axes(ax, 1.3, loc=3) # zoom = 6
axins.imshow(Z2, extent=extent, interpolation="nearest",
              origin="upper",  cmap="gray", vmin=0,vmax=255)

# sub region of the original image
x1, x2, y1, y2 =131, 197, 125, 191
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)

plt.axis('off')
plt.grid(b=None)


mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.draw()
plt.savefig("parrot.png", bbox_inches='tight', pad_inches=0)
plt.show()

"""
"""
Guardado de imagenes sin marco Binary

CS = np.load('CS_Binary_50.npy')
pseudo_inv=np.linalg.pinv(CS, rcond=1e-15, hermitian=False)
   

Iorg, row, col, Ipad, row_new, col_new, block_size=imread_CS_py(0)
y_im_crop, block_num, y_train2=img2col_py(Ipad, block_size,CS)
model = models.load_model('RECON_NET_88912_Binary50_V0.h5')
prediction = model.predict(y_train2)
X_rec=col2im_CS_py(prediction.transpose(), row, col, row_new, col_new)
    
plt.imshow(X_rec, cmap="gray", vmin=0,vmax=255)
plt.axis('off')
plt.grid(b=None)
plt.savefig("monarca-full.png", bbox_inches='tight')
plt.show()
X_rec = Image.fromarray(X_rec)
X_rec2 = X_rec.crop((132,132,196,196))
np.array(X_rec2)
plt.imshow(X_rec2, cmap="gray", vmin=0,vmax=255)
plt.axis('off')
plt.grid(b=None)
plt.savefig("monarca-crop.png", bbox_inches='tight')
plt.show()


Iorg, row, col, Ipad, row_new, col_new, block_size=imread_CS_py(4)
y_im_crop, block_num, y_train2=img2col_py(Ipad, block_size,CS)
model = models.load_model('RECON_NET_88912_Binary50_V0.h5')
prediction = model.predict(y_train2)
X_rec=col2im_CS_py(prediction.transpose(), row, col, row_new, col_new)
plt.imshow(X_rec, cmap="gray", vmin=0,vmax=255)
plt.axis('off')
plt.grid(b=None)
plt.savefig("parrot-full.png", bbox_inches='tight')
plt.show()
X_rec = Image.fromarray(X_rec)
X_rec2 = X_rec.crop((132,66,196,130))
np.array(X_rec2)
plt.imshow(X_rec2, cmap="gray", vmin=0,vmax=255)
plt.axis('off')
plt.grid(b=None)
plt.savefig("parrot-crop.png", bbox_inches='tight')
plt.show()

"""

