#This simple script prepares data for CNN

########################################    Import libraries    #############################################
import csv
import argparse
import math
import ast
import numpy as np
import logging

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser(description='This script helps to visualise the seeds by projecting their hit coordinates to the 2-d screen.')
parser.add_argument('--Res',help="Please enter the scaling resolution in microns", default='100')
parser.add_argument('--StartImage',help="Please select the beginning Image", default='1')
parser.add_argument('--Images',help="Please select the number of Images", default='1')
parser.add_argument('--Title',help="Plot title", default='')
parser.add_argument('--PlotType',help="Enter plot type: XZ/YZ/3D", default='XZ')
parser.add_argument('--Predict',help="Use the CNN model to predict the seed : Y/N ?", default='N')
parser.add_argument('--Type',help="Please enter the sample type: Val or number for the training set", default='1')
parser.add_argument('--Label',help="Which labels would you like to see: 'ANY/Fake/Truth", default='ANY')
parser.add_argument('--ModelName',help="Name of the model", default='2T_100_MC_1_model')
########################################     Main body functions    #########################################
args = parser.parse_args()
SeedNo=int(args.Images)
resolution=float(args.Res)
StartImage=int(args.StartImage)
if StartImage>SeedNo:
    SeedNo=StartImage

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
import Parameters as PM

MaxX=PM.MaxX
MaxY=PM.MaxY
MaxZ=PM.MaxZ

boundsX=int(round(MaxX/resolution,0))
boundsY=int(round(MaxY/resolution,0))
boundsZ=int(round(MaxZ/resolution,0))
H=(boundsX)*2
W=(boundsY)*2
L=(boundsZ)

print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising EDER-VIANN image visualisation module  #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
if args.Type=='Val':
 input_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_VALIDATION_SET.csv'
else:
 input_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_TRAIN_SET_'+args.Type+'.csv'
#print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
if args.PlotType=='XZ':
  InitialData=[]
  Index=-1
  for x in range(-boundsX,boundsX):
          for z in range(0,boundsZ):
            InitialData.append(0.0)
  Matrix = np.array(InitialData)
  Matrix=np.reshape(Matrix,(H,L))
if args.PlotType=='YZ':
 InitialData=[]
 Index=-1
 for y in range(-boundsY,boundsY):
          for z in range(0,boundsZ):
            InitialData.append(0.0)

 Matrix = np.array(InitialData)
 Matrix=np.reshape(Matrix,(W,L))
if args.PlotType=='XY':
  InitialData=[]
  Index=-1
  for x in range(-boundsX,boundsX):
          for y in range(-boundsY,boundsY):
            InitialData.append(0.0)
  Matrix = np.array(InitialData)
  Matrix=np.reshape(Matrix,(H,W))
for sd in range(StartImage-1,SeedNo):
 progress=int( round( (float(sd)/float(SeedNo)*100),1)  )
 print("Loading data, progress is ",progress, end="\r", flush=True)
 #Locate mothers
 data=UF.LoadImage(input_file_location,sd,args.Label)
 if data==[] and (SeedNo==StartImage):
    print(UF.TimeStamp(), bcolors.FAIL+"Do not have the required image with a given characteristics..."+bcolors.ENDC)
    exit()
 elif data==[]:
    continue
 elif type(data) is int:
    print(UF.TimeStamp(), bcolors.FAIL+"Your StartImage number",StartImage, "exceeds total number of images in the given set, which is",str(data), "images"+bcolors.ENDC)
    exit()
 elif args.Predict=='Y' and data[2][0]==0 and StartImage>=SeedNo:
     import os
     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
     logging.getLogger('tensorflow').setLevel(logging.FATAL)
     import warnings
     warnings.simplefilter(action='ignore', category=FutureWarning)
     import tensorflow as tf
     from tensorflow import keras
     model_name=EOS_DIR+'/EDER-VIANN/Models/'+args.ModelName
     model=tf.keras.models.load_model(model_name)
     SeedImage=UF.LoadRenderImages([data],resolution,MaxX,MaxY,MaxZ,1,1,False)[0]
     pred = model.predict(SeedImage)
     Title='Fake Vertex Image || The predicted vertex probability is '+str(round(float(pred[0][1])*100,2))+' %'
 elif args.Predict=='Y' and data[2][0]==1 and StartImage>=SeedNo:
     import os
     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
     logging.getLogger('tensorflow').setLevel(logging.FATAL)
     import warnings
     warnings.simplefilter(action='ignore', category=FutureWarning)
     import tensorflow as tf
     from tensorflow import keras
     model_name=EOS_DIR+'/EDER-VIANN/Models/'+args.ModelName
     model=tf.keras.models.load_model(model_name)
     SeedImage=UF.LoadRenderImages([data],resolution,MaxX,MaxY,MaxZ,1,1,False)[0]
     pred = model.predict(SeedImage)
     Title='Truth Vertex Image || The predicted vertex probability is '+str(round(float(pred[0][1])*100,2))+' %'
 elif args.Title=='' and args.Label!='ANY':
    Title=args.Label+' Vertex Image'
 elif args.Title=='' and args.Label=='ANY':
    if data[2][0]==0:

        Title='Fake Vertex Image'
    else:
        Title='Truth Vertex Image'

 else:
    Title=args.Title
 additional_data=[]
 additional_data=UF.EnrichImage(resolution, data)
 additional_data=UF.ChangeImageResoluion(resolution,additional_data)
 data=UF.ChangeImageResoluion(resolution, data)
 if args.PlotType=='XZ':
  for Tracks in data[1]:
           for Hits in Tracks:
                if abs(Hits[0])<boundsX and abs(Hits[2])<boundsZ:
                   Matrix[Hits[0]+boundsX][Hits[2]]+=1

  for Tracks in additional_data[1]:
      for Hits in Tracks:
        if abs(Hits[0])<boundsX and abs(Hits[2])<boundsZ:
          Matrix[Hits[0]+boundsX][Hits[2]]+=1
 if args.PlotType=='YZ':

  for Tracks in data[1]:
         for Hits in Tracks:
                 if abs(Hits[1])<boundsY and abs(Hits[2])<boundsZ:
                   Matrix[Hits[1]+boundsY][Hits[2]]+=1
  for Tracks in additional_data[1]:
      for Hits in Tracks:
         if abs(Hits[1])<boundsY and abs(Hits[2])<boundsZ:
          Matrix[Hits[1]+boundsY][Hits[2]]+=1
 if args.PlotType=='XY':
  for Tracks in data[1]:
     for Hits in Tracks:
       if abs(Hits[0])<boundsX and abs(Hits[1])<boundsY:
         Matrix[Hits[0]+boundsX][Hits[1]+boundsY]+=1
  for Tracks in additional_data[1]:
      for Hits in Tracks:
         if abs(Hits[0])<boundsX and abs(Hits[1])<boundsY:
          Matrix[Hits[0]+boundsX][Hits[1]+boundsY]+=1
import matplotlib as plt
from matplotlib.colors import LogNorm
if args.PlotType=='XZ':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('Z [microns /'+str(int(resolution))+']')
 plt.ylabel('X [microns /'+str(int(resolution))+']')

 image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsX,-boundsX],norm=LogNorm())
 plt.gca().invert_yaxis()
 plt.show()
if args.PlotType=='YZ':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('Z [microns /'+str(int(resolution))+']')
 plt.ylabel('Y [microns /'+str(int(resolution))+']')
 image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsY,-boundsY],norm=LogNorm())
 plt.gca().invert_yaxis()
 plt.show()
if args.PlotType=='XY':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('X [microns /'+str(int(resolution))+']')
 plt.ylabel('Y [microns /'+str(int(resolution))+']')
 image=plt.imshow(Matrix,cmap='gray_r',extent=[boundsX,-boundsX,-boundsY,boundsY],norm=LogNorm())
 plt.gca().invert_xaxis()
 plt.show()
exit()




