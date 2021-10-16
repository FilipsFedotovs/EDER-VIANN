#This simple script prepares data for CNN
########################################    Import libraries    #############################################
#import csv
import Utility_Functions as UF
import argparse
import pandas as pd #We use Panda for a routine data processing
import os, psutil #helps to monitor the memory
import gc  #Helps to clear memory
from Utility_Functions import Seed
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
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Set',help="Set Number", default='1')
parser.add_argument('--SubSet',help="SubSet Number", default='1')
parser.add_argument('--Fraction',help="Fraction", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--VO_T',help="The maximum distance between vertex reconstructed origin and the start hit of either tracks", default='3900')
parser.add_argument('--VO_min_Z',help="Minimal Z coordinate of the reconstructed vertex origin", default='-39500')
parser.add_argument('--VO_max_Z',help="Maximum Z coordinate of the reconstructed vertex origin", default='0')
parser.add_argument('--MaxDoca',help="Doca cut in microns", default='200')
parser.add_argument('--MinAngle',help="Minimal angle in radians", default='0.0')
parser.add_argument('--MaxAngle',help="Maximum angle in radians", default='2.0')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
SubSet=args.SubSet
fraction=args.Fraction
MaxDoca=float(args.MaxDoca)
VO_min_Z=float(args.VO_min_Z)
VO_max_Z=float(args.VO_max_Z)
VO_T=float(args.VO_T)
MinAngle=float(args.MinAngle)
MaxAngle=float(args.MaxAngle)

#Converting image size bounds in line with resolution settings
boundsX=int(round(MaxX/resolution,0))
boundsY=int(round(MaxY/resolution,0))
boundsZ=int(round(MaxZ/resolution,0))
H=boundsX*2
W=boundsY*2
L=boundsZ
AFS_DIR=args.AFS
EOS_DIR=args.EOS

input_track_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M1_TRACKS.csv'
input_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M2_M3_RawSeeds_'+Set+'_'+SubSet+'_'+fraction+'.csv'
output_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M3_RawImages_'+Set+'_'+SubSet+'_'+fraction+'.pkl'
print(UF.TimeStamp(),'Loading the data')
seeds=pd.read_csv(input_seed_file_location)
seeds_1=seeds.drop(['Track_2'],axis=1)
seeds_1=seeds_1.rename(columns={"Track_1": "Track_ID"})
seeds_2=seeds.drop(['Track_1'],axis=1)
seeds_2=seeds_2.rename(columns={"Track_2": "Track_ID"})
seed_list=result = pd.concat([seeds_1,seeds_2])
seed_list=seed_list.sort_values(['Track_ID'])
seed_list.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
tracks=pd.read_csv(input_track_file_location)
print(UF.TimeStamp(),'Analysing the data')
tracks=pd.merge(tracks, seed_list, how="inner", on=["Track_ID"]) #Shrinking the Track data so just a star hit for each track is present.
tracks["x"] = pd.to_numeric(tracks["x"],downcast='float')
tracks["y"] = pd.to_numeric(tracks["y"],downcast='float')
tracks["z"] = pd.to_numeric(tracks["z"],downcast='float')
tracks = tracks.values.tolist() #Convirting the result to List data type
seeds = seeds.values.tolist() #Convirting the result to List data type
del seeds_1
del seeds_2
del seed_list
gc.collect()
limit=len(seeds)
seed_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
#create seeds
GoodSeeds=[]
print(UF.TimeStamp(),'Beginning the image generation part...')
for s in range(0,limit):
    seed=seeds.pop(0)
    label=seed[2]
    seed=Seed(seed[:2])
    if label:
        num_label = 1
    else:
        num_label = 0
    seed.MCtruthClassifySeed(num_label)
    seed.DecorateTracks(tracks)
    try:
      seed.DecorateSeedGeoInfo()
    except:
      continue
    seed.SeedQualityCheck(VO_min_Z,VO_max_Z,MaxDoca,VO_T,MinAngle,MaxAngle)
    if seed.GeoFit:
           GoodSeeds.append(seed)
    else:
        del seed
        continue
print(UF.TimeStamp(),bcolors.OKGREEN+'The raw image generation has been completed..'+bcolors.ENDC)
del tracks
del seeds
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
open_file = open(output_seed_file_location, "wb")
pickle.dump(GoodSeeds, open_file)
open_file.close()
exit()
