#This simple script is design for studying and selecting an optimal size and resolution of the images

########################################    Import libraries    #############################################
import csv
import argparse
import math
import ast
import numpy as np
import logging
import pickle

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
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
from Utility_Functions import Seed
import Parameters as PM

parser = argparse.ArgumentParser(description='This script helps to visualise the seeds by projecting their hit coordinates to the 2-d screen.')
parser.add_argument('--Res',help="Please enter the scaling resolution in microns", default=PM.resolution)
parser.add_argument('--StartImage',help="Please select the beginning Image", default='1')
parser.add_argument('--Images',help="Please select the number of Images", default='1')
parser.add_argument('--PlotType',help="Enter plot type: XZ/YZ/3D", default='XZ')
parser.add_argument('--MaxX',help="Enter max half height of the image in microns", default=PM.MaxX)
parser.add_argument('--MaxY',help="Enter max half width of the image in microns", default=PM.MaxY)
parser.add_argument('--MaxZ',help="Enter max length of the image in microns", default=PM.MaxZ)
parser.add_argument('--Rescale',help="Rescale the images : Y/N ?", default='N')
parser.add_argument('--Type',help="Please enter the sample type: Val or number for the training set", default='1')
parser.add_argument('--Label',help="Which labels would you like to see: 'ANY/Fake/Truth", default='ANY')
########################################     Main body functions    #########################################
args = parser.parse_args()
SeedNo=int(args.Images)
resolution=float(args.Res)
MaxX=float(args.MaxX)
MaxY=float(args.MaxY)
MaxZ=float(args.MaxZ)
StartImage=int(args.StartImage)
if StartImage>SeedNo:
    SeedNo=StartImage
if args.Rescale=='Y':
    Rescale=True
else:
    Rescale=False
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
 input_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M4_Validation_Set.pkl'
else:
 input_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M4_Train_Set_'+args.Type+'.pkl'


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






#Locate mothers
data_file=open(input_file_location,'rb')
data=pickle.load(data_file)
data=data[StartImage-1:min(SeedNo,len(data))]
data_file.close()


Title=args.Label+' Vertex Image'

if args.Label=='Truth':
     data=[im for im in data if im.MC_truth_label == 1]
if args.Label=='Fake':
     data=[im for im in data if im.MC_truth_label == 0]
counter=0
for sd in data:
 progress=int( round( (float(counter)/float(len(data))*100),1)  )
 print('Rendering images, progress is ',progress, end="\r", flush=True)
 counter+=1
 sd.PrepareTrackPrint(MaxX,MaxY,MaxZ,resolution,Rescale)
 if args.PlotType=='XZ':
  for Hits in sd.TrackPrint:
      if abs(Hits[0])<boundsX and abs(Hits[2])<boundsZ:
                   Matrix[Hits[0]+boundsX][Hits[2]]+=1
 if args.PlotType=='YZ':
        for Hits in sd.TrackPrint:
                 if abs(Hits[1])<boundsY and abs(Hits[2])<boundsZ:
                   Matrix[Hits[1]+boundsY][Hits[2]]+=1
 if args.PlotType=='XY':
     for Hits in sd.TrackPrint:
       if abs(Hits[0])<boundsX and abs(Hits[1])<boundsY:
         Matrix[Hits[0]+boundsX][Hits[1]+boundsY]+=1
image_no=len(data)
del data
import matplotlib as plt
from matplotlib.colors import LogNorm
if args.PlotType=='XZ':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('Z [microns /'+str(int(resolution))+']')
 plt.ylabel('X [microns /'+str(int(resolution))+']')

 if image_no==1:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsX,-boundsX])#,norm=LogNorm())
 else:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsX,-boundsX],norm=LogNorm())
 plt.gca().invert_yaxis()
 plt.show()
if args.PlotType=='YZ':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('Z [microns /'+str(int(resolution))+']')
 plt.ylabel('Y [microns /'+str(int(resolution))+']')
 if image_no==1:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsY,-boundsY])#,norm=LogNorm())
 else:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[0,boundsZ,boundsY,-boundsY],norm=LogNorm())
 plt.gca().invert_yaxis()
 plt.show()
if args.PlotType=='XY':
 import numpy as np
 from matplotlib import pyplot as plt
 plt.title(Title)
 plt.xlabel('X [microns /'+str(int(resolution))+']')
 plt.ylabel('Y [microns /'+str(int(resolution))+']')
 if image_no==1:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[boundsX,-boundsX,-boundsY,boundsY])#,norm=LogNorm())
 else:
    image=plt.imshow(Matrix,cmap='gray_r',extent=[boundsX,-boundsX,-boundsY,boundsY],norm=LogNorm())
 plt.gca().invert_xaxis()
 plt.show()
exit()




