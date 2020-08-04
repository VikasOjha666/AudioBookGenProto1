import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import librosa
from gtts import gTTS
import pytesseract
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"





def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs,conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    

    if(pretrained_weights):
      model.load_weights(pretrained_weights)

    return model

model=unet()
model.load_weights('./text_seg_model.h5')



def generate_audios(filename):
    
    file=open('recognized_texts.txt','w')
    img=cv2.imread(f'{filename}',0)
    ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    img=cv2.resize(img,(512,512))
   
    img= np.expand_dims(img,axis=-1)

    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)
    pred=np.squeeze(np.squeeze(pred,axis=0),axis=-1)

    plt.imsave('test_img_mask.JPG',pred)

    coordinates=[]
    img = cv2.imread('test_img_mask.JPG',0) 
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    ori_img=cv2.imread(f'{filename}')
    #ori_img=cv2.resize(ori_img,(512,512))

    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)
    
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        #cv2.rectangle(ori_img, (x, y), (x+w,y+h), 255, 1)
        coordinates.append((int(x*rW),int(y*rH),int((x+w)*rW),int((y+h)*rH)))

        
    
    for i in range(len(coordinates)-1,-1,-1):
        coors=coordinates[i]

        p_img=ori_img[coors[1]:coors[3],coors[0]:coors[2]].copy()
        file.writelines(pytesseract.image_to_string(p_img)+'\n')

    file.close()

    file=open('recognized_texts.txt','r').readlines()

   

    sounds=[]

    for line in file:
        if line!='':
            try:
                tts=gTTS(text=str(line),lang='en-IN')
                tts.save('1.mp3')
                y,sr=librosa.load('1.mp3')
                sounds.append(y)
            except:
                pass
            

    joined=np.array([])
    os.remove('./1.mp3')

    for i in range(len(sounds)):
        joined=np.append(joined,sounds[i])

    filename=filename.split('/')[-1].split('.')[0]

    librosa.output.write_wav(f'./Audios/{filename}.wav',joined,22050)





    


