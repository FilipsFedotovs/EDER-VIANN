#This simple script prepares data for CNN

########################################    Import libraries    #############################################
import csv
import argparse
import math
import ast
import numpy as np
import tensorflow as tf
import copy
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Res',help="Please enter the scaling resolution in microns", default='1000')
parser.add_argument('--Mode',help="Please enter the mode: Production/Test/Evolution", default='Test')
parser.add_argument('--ImageSet',help="Please enter the image set", default='1')
########################################     Main body functions    #########################################
args = parser.parse_args()
ImageSet=args.ImageSet
resolution=float(args.Res)
MaxX=10000.0
MaxY=10000.0
MaxZ=20000.0
boundsX=int(round(MaxX/resolution,0))
boundsY=int(round(MaxY/resolution,0))
boundsZ=int(round(MaxZ/resolution,0))
H=boundsX*2
W=boundsY*2
L=boundsZ

#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF

flocation=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/'+'CNN_TRAIN_IMAGES_'+ImageSet+'.csv'
vlocation=EOS_DIR+'/EDER-VIANN/Data/VALIDATION_SET/'+'CNN_VALIDATION_IMAGES_1.csv'

print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising EDER-VIANN image model creation module #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
TrainImages=UF.LoadImages(flocation)
print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been loaded successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+vlocation+bcolors.ENDC)
ValidationImages=UF.LoadImages(vlocation)
print(UF.TimeStamp(), bcolors.OKGREEN+"Validation data has been loaded successfully..."+bcolors.ENDC)


##########################################################   Data enrichment (Filling gaps between tracklets)   ##############################
print(UF.TimeStamp(),'Enriching training data...')
additional_train_data=[]
for TI in TrainImages:
  additional_train_data.append(UF.EnrichImage(resolution, TI))
print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been enriched successfully..."+bcolors.ENDC)

print(UF.TimeStamp(),'Enriching validation data...')
additional_val_data=[]
for VI in ValidationImages:
  additional_val_data.append(UF.EnrichImage(resolution, VI))
print(UF.TimeStamp(), bcolors.OKGREEN+"Validation data has been enriched successfully..."+bcolors.ENDC)


########################################################  Image preparation for rendering   ########################################
print(UF.TimeStamp(),'Rendering train images...')
TrainImagesY=np.empty([len(TrainImages),1])
TrainImagesX=np.empty([len(TrainImages),H,W,L])
for TI in range(0,len(TrainImages)):
    TrainImages[TI]=UF.ChangeImageResoluion(resolution, TrainImages[TI])
    additional_train_data[TI]=UF.ChangeImageResoluion(resolution, additional_train_data[TI])
    progress=int(round((float(TI)/float(len(TrainImages)))*100,0))
    print("Progress is ",progress,' %', end="\r", flush=True)
    TrainImagesY[TI]=int(2*float(TrainImages[TI][3]))
    BlankRenderedTrainImage=[]
    for x in range(-boundsX,boundsX):
          for y in range(-boundsY,boundsY):
            for z in range(0,boundsZ):
             BlankRenderedTrainImage.append(0.1)
    RenderedTrainImage = np.array(BlankRenderedTrainImage)
    RenderedTrainImage = np.reshape(RenderedTrainImage,(H,W,L))
    for Tracks in TrainImages[TI][4]:
     for Hits in Tracks:
         if abs(Hits[0])<boundsX and abs(Hits[1])<boundsX and abs(Hits[2])<boundsZ:
             RenderedTrainImage[Hits[0]+boundsX][Hits[1]+boundsY][Hits[2]]=0.99
    for Tracks in additional_train_data[TI][4]:
     for Hits in Tracks:
       if abs(Hits[0])<boundsX and abs(Hits[1])<boundsX and abs(Hits[2])<boundsZ:
         RenderedTrainImage[Hits[0]+boundsX][Hits[1]+boundsY][Hits[2]]=0.99
    TrainImagesX[TI]=RenderedTrainImage
TrainImagesX= TrainImagesX[..., np.newaxis]
TrainImagesY=tf.keras.utils.to_categorical(TrainImagesY)
print(UF.TimeStamp(), bcolors.OKGREEN+"Train images have been rendered successfully..."+bcolors.ENDC)

print(UF.TimeStamp(),'Loading the model...')
##### This but has to be converted to a part that interprets DNA code  ###################################
model = Sequential()
model.add(Conv3D(32, activation='relu',kernel_size=(3,3,3),kernel_initializer='he_uniform', input_shape=(H,W,L,1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(64, activation='relu',kernel_size=(3,3,3),kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(TrainImagesY.shape[1], activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
###################################################################################################

print(UF.TimeStamp(),'Training the model...')
# Fit data to model
history = model.fit(TrainImagesX, TrainImagesY,batch_size=32,epochs=20,verbose=1)

########################################################  Image preparation for rendering   ########################################
print(UF.TimeStamp(),'Rendering validation images...')
ValImagesY=[]
ValImagesX=np.empty([len(ValidationImages),H,W,L])
for VI in range(0,len(ValidationImages)):
    ValidationImages[VI]=UF.ChangeImageResoluion(resolution, ValidationImages[VI])
    additional_val_data[VI]=UF.ChangeImageResoluion(resolution, additional_val_data[VI])
    progress=int(round((float(VI)/float(len(ValidationImages)))*100,0))
    print("Progress is ",progress,' %', end="\r", flush=True)
    ValImagesY.append(int(2*float(ValidationImages[VI][3])))
    BlankRenderedValImage=[]
    for x in range(-boundsX,boundsX):
          for y in range(-boundsY,boundsY):
            for z in range(0,boundsZ):
             BlankRenderedValImage.append(0.1)
    RenderedValImage = np.array(BlankRenderedValImage)
    RenderedValImage = np.reshape(RenderedValImage,(H,W,L))
    for Tracks in ValidationImages[VI][4]:
     for Hits in Tracks:
         if abs(Hits[0])<boundsX and abs(Hits[1])<boundsX and abs(Hits[2])<boundsZ:
             RenderedValImage[Hits[0]+boundsX][Hits[1]+boundsY][Hits[2]]=0.99
    for Tracks in additional_val_data[VI][4]:
     for Hits in Tracks:
       if abs(Hits[0])<boundsX and abs(Hits[1])<boundsX and abs(Hits[2])<boundsZ:
         RenderedValImage[Hits[0]+boundsX][Hits[1]+boundsY][Hits[2]]=0.99
    ValImagesX[VI]=RenderedValImage
ValImagesX= ValImagesX[..., np.newaxis]
print(UF.TimeStamp(), bcolors.OKGREEN+"Validation images have been rendered successfully..."+bcolors.ENDC)

print(UF.TimeStamp(),'Validating the model...')
pred = model.predict(ValImagesX)
pred = np.argmax(pred, axis=1)
match=0
for p in range(0,len(pred)):
      if int(ValImagesY[p])==pred[p]:
            match+=1
Accuracy=int(round((float(match)/float(len(pred)))*100,0))
print('Overall accuracy of the model is',Accuracy,'%')
VertexLengths=[]
for VR in ValImagesY:
    if (VR in VertexLengths)==False:
          VertexLengths.append(VR)
print(ValImagesY)
print(VertexLengths)
print(pred)


for VC in VertexLengths:
   overall_match=0
   hit_match=0
   for VI in range(0,len(ValImagesY)):
     if  int(ValImagesY[VI])==int(VC):
         overall_match+=1
         if int(ValImagesY[VI])==int(pred[VI]):
            hit_match+=1
   Accuracy=int(round((float(hit_match)/float(overall_match))*100,0))
   print('------------------------------------------------------------------')
   print('The accuracy of the model for',str(float(round((VC/2.0),1))),'-track vertices is',Accuracy,'%')


exit()


