#This simple script prepares data for CNN
########################################    Import libraries    #############################################
#import csv
import Utility_Functions as UF
import argparse
import pandas as pd #We use Panda for a routine data processing
import pickle
import tensorflow as tf
from tensorflow import keras

import os, psutil #helps to monitor the memory
import gc  #Helps to clear memory

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
parser.add_argument('--Set',help="Set Number", default='1')
parser.add_argument('--Fraction',help="Fraction", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--resolution',help="Resolution in microns per pixel", default='100')
parser.add_argument('--acceptance',help="Vertex fit minimum acceptance", default='0.5')
parser.add_argument('--MaxX',help="Image size in microns along the x-axis", default='3500.0')
parser.add_argument('--MaxY',help="Image size in microns along the y-axis", default='1000.0')
parser.add_argument('--MaxZ',help="Image size in microns along the z-axis", default='20000.0')
parser.add_argument('--ModelName',help="Name of the CNN model", default='2T_100_MC_1_model')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
fraction=str(int(args.Fraction)+1)
resolution=float(args.resolution)
acceptance=float(args.acceptance)
#Maximum bounds on the image size in microns
MaxX=float(args.MaxX)
MaxY=float(args.MaxY)
MaxZ=float(args.MaxZ)
#Converting image size bounds in line with resolution settings
AFS_DIR=args.AFS
EOS_DIR=args.EOS
input_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R3_R4_FilteredSeeds_'+Set+'_'+fraction+'.pkl'
output_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R4_R4_RecSeeds_'+Set+'_'+fraction+'.pkl'
print(UF.TimeStamp(),'Analysing the data')
seeds_file=open(input_seed_file_location,'rb')
seeds=pickle.load(seeds_file)
seeds_file.close()
limit=len(seeds)
seed_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
print(UF.TimeStamp(),'Loading the model...')
#Load the model
model_name=EOS_DIR+'/EDER-VIANN/Models/'+args.ModelName
model=tf.keras.models.load_model(model_name)
#create seeds
GoodSeeds=[]
print(UF.TimeStamp(),'Beginning the vertexing part...')
for s in range(0,limit):
    seed=seeds.pop(0)
    seed.PrepareTrackPrint(MaxX,MaxY,MaxZ,resolution,True)
    SeedImage=UF.LoadRenderImages([seed],1,1)[0]
    seed.UnloadTrackPrint()
    seed.CNNFitSeed(model.predict(SeedImage)[0][1])
    if seed.Seed_CNN_Fit>=acceptance:
              GoodSeeds.append(seed)
    else:
              continue
print(UF.TimeStamp(),bcolors.OKGREEN+'The vertexing has been completed..'+bcolors.ENDC)
del seeds
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
open_file = open(output_seed_file_location, "wb")
pickle.dump(GoodSeeds, open_file)
open_file.close()
exit()
