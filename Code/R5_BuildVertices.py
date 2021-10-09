#This simple merges 2-Track vettices to produce the final result
# Part of EDER-VIANN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import math #We use it for data manipulation
import gc  #Helps to clear memory
import numpy as np
import os
import pickle
import random


class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script takes vertex-fitted 2-track seed candidates from previous step and merges them if seeds have a common track')
parser.add_argument('--Acceptance',help="Minimum acceptance for the track", default='0.5')
parser.add_argument('--DataCut',help="In how many chunks would you like to split data?", default='30')
parser.add_argument('--Mode',help="Restart (R) or Continue (C)", default='C')
######################################## Set variables  #############################################################
args = parser.parse_args()

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
import Utility_Functions as UF #This is where we keep routine utility functions
from Utility_Functions import Seed
import Parameters as PM #This is where we keep framework global parameters
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-VIANN Vertex  module             ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
if args.Mode=='R':
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R5', ['R5_R5'], "SoftUsed == \"EDER-VIANN-R5\"")
    Acceptance=float(args.Acceptance)
if args.Mode=='R':
    print(UF.TimeStamp(), "Starting the script from the scratch")
    input_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R4_R5_Rec_Seeds.pkl'
    print(UF.TimeStamp(), "Loading vertexed seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data_file=open(input_file_location,'rb')
    base_data=pickle.load(data_file)
    data_file.close()
    print(UF.TimeStamp(), bcolors.OKGREEN+"Loading is successful, there are "+str(len(base_data))+" vertexed seeds..."+bcolors.ENDC)
    print(UF.TimeStamp(), "Stripping off the seeds with low acceptance...")
    base_data=[sd for sd in base_data if sd.Seed_CNN_Fit >= Acceptance]
    print(UF.TimeStamp(), bcolors.OKGREEN+"The refining was successful, "+str(len(base_data))+" seeds remain..."+bcolors.ENDC)
    output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Rec_Seeds_Refined.pkl'
    open_file = open(output_file_location, "wb")
    pickle.dump(base_data, open_file)
    open_file.close()
    no_iter=int(math.ceil(float(len(base_data)/float(MaxSeedsPerVxPool))))
    print(no_itr)
#     for i in range(0,int(args.DataCut)):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds_'+str(i)+'.pkl'
#                   if os.path.isfile(output_file_location)==False:
#                       print(UF.TimeStamp(), "Analysing set",i+1)
#                       length=len(RefinedData)
#                       segment=int(round(math.ceil(length/int(args.DataCut)),0))
#                       NewData=RefinedData[(i*segment):((i+1)*segment)]
#                       print(UF.TimeStamp(), "Initiating the seed merging...")
#                       InitialDataLength=len(NewData)
#                       SeedCounter=0
#                       SeedCounterContinue=True
#                       while SeedCounterContinue:
#                           if SeedCounter>=len(NewData):
#                               SeedCounterContinue=False
#                               break
#                           progress=round(float(SeedCounter)/float(len(NewData))*100,0)
#                           print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
#                           SubjectSeed=NewData[SeedCounter]
#                           for ObjectSeed in NewData[SeedCounter+1:]:
#                               if SubjectSeed.InjectSeed(ObjectSeed):
#                                    NewData.pop(NewData.index(ObjectSeed))
#                           SeedCounter+=1
#                       print(str(InitialDataLength), "2-track vertices were merged into", str(len(NewData)), 'vertices with higher multiplicity...')
#
#                       open_file = open(output_file_location, "wb")
#                       pickle.dump(NewData, open_file)
#                       open_file.close()
#
#     del RefinedData
#     gc.collect()
#     status=1
#
# if status==1:
#     print(UF.TimeStamp(),'Current stage is no:', status)
#     VertexPool=[]
#     for i in range(0,int(args.DataCut)):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds_'+str(i)+'.pkl'
#                   data_file=open(output_file_location,'rb')
#                   NewData=pickle.load(data_file)
#                   data_file.close()
#                   VertexPool+=NewData
#                   random.shuffle(VertexPool)
#     for i in range(0,int(round(float(args.DataCut)*0.6,0))):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds2_'+str(i)+'.pkl'
#                   if os.path.isfile(output_file_location)==False:
#                       print(UF.TimeStamp(), "Analysing set",i+1)
#                       length=len(VertexPool)
#                       segment=int(round(math.ceil(length/int(round(float(args.DataCut)*0.6,0))),0))
#                       NewData=VertexPool[(i*segment):((i+1)*segment)]
#                       print(UF.TimeStamp(), "Initiating the seed merging...")
#                       InitialDataLength=len(NewData)
#                       SeedCounter=0
#                       SeedCounterContinue=True
#                       while SeedCounterContinue:
#                           if SeedCounter>=len(NewData):
#                               SeedCounterContinue=False
#                               break
#                           progress=round(float(SeedCounter)/float(len(NewData))*100,0)
#                           print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
#                           SubjectSeed=NewData[SeedCounter]
#                           for ObjectSeed in NewData[SeedCounter+1:]:
#                               if SubjectSeed.InjectSeed(ObjectSeed):
#                                    NewData.pop(NewData.index(ObjectSeed))
#                           SeedCounter+=1
#                       print(str(InitialDataLength), "2-track vertices were merged into", str(len(NewData)), 'vertices with higher multiplicity...')
#
#                       open_file = open(output_file_location, "wb")
#                       pickle.dump(NewData, open_file)
#                       open_file.close()
#     print(UF.TimeStamp(), "Initiating the global vertex merging...")
#     del VertexPool
#     gc.collect()
#     status=2
#
# if status==2:
#     print(UF.TimeStamp(),'Current stage is no:', status)
#     VertexPool=[]
#     for i in range(0,int(round(float(args.DataCut)*0.6,0))):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds2_'+str(i)+'.pkl'
#                   data_file=open(output_file_location,'rb')
#                   NewData=pickle.load(data_file)
#                   data_file.close()
#                   VertexPool+=NewData
#                   random.shuffle(VertexPool)
#
#     for i in range(0,int(round(float(args.DataCut)*0.6*0.6,0))):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds3_'+str(i)+'.pkl'
#                   if os.path.isfile(output_file_location)==False:
#                       print(UF.TimeStamp(), "Analysing set",i+1)
#                       length=len(VertexPool)
#                       segment=int(round(math.ceil(length/int(round(float(args.DataCut)*0.6*0.6,0))),0))
#                       NewData=VertexPool[(i*segment):((i+1)*segment)]
#                       print(UF.TimeStamp(), "Initiating the seed merging...")
#                       InitialDataLength=len(NewData)
#                       SeedCounter=0
#                       SeedCounterContinue=True
#                       while SeedCounterContinue:
#                           if SeedCounter>=len(NewData):
#                               SeedCounterContinue=False
#                               break
#                           progress=round(float(SeedCounter)/float(len(NewData))*100,0)
#                           print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
#                           SubjectSeed=NewData[SeedCounter]
#                           for ObjectSeed in NewData[SeedCounter+1:]:
#                               if SubjectSeed.InjectSeed(ObjectSeed):
#                                    NewData.pop(NewData.index(ObjectSeed))
#                           SeedCounter+=1
#                       print(str(InitialDataLength), "2-track vertices were merged into", str(len(NewData)), 'vertices with higher multiplicity...')
#
#                       open_file = open(output_file_location, "wb")
#                       pickle.dump(NewData, open_file)
#                       open_file.close()
#     del VertexPool
#     gc.collect()
#     status=3
#
# if status==3:
#     print(UF.TimeStamp(),'Current stage is no:', status)
#     VertexPool=[]
#     for i in range(0,int(round(float(args.DataCut)*0.6*0.6,0))):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds3_'+str(i)+'.pkl'
#                   data_file=open(output_file_location,'rb')
#                   NewData=pickle.load(data_file)
#                   data_file.close()
#                   VertexPool+=NewData
#                   random.shuffle(VertexPool)
#
#     for i in range(0,int(round(float(args.DataCut)*0.6*0.6*0.6,0))):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds4_'+str(i)+'.pkl'
#                   if os.path.isfile(output_file_location)==False:
#                       print(UF.TimeStamp(), "Analysing set",i+1)
#                       length=len(VertexPool)
#                       segment=int(round(math.ceil(length/int(round(float(args.DataCut)*0.6*0.6*0.6,0))),0))
#                       NewData=VertexPool[(i*segment):((i+1)*segment)]
#                       print(UF.TimeStamp(), "Initiating the seed merging...")
#                       InitialDataLength=len(NewData)
#                       SeedCounter=0
#                       SeedCounterContinue=True
#                       while SeedCounterContinue:
#                           if SeedCounter>=len(NewData):
#                               SeedCounterContinue=False
#                               break
#                           progress=round(float(SeedCounter)/float(len(NewData))*100,0)
#                           print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
#                           SubjectSeed=NewData[SeedCounter]
#                           for ObjectSeed in NewData[SeedCounter+1:]:
#                               if SubjectSeed.InjectSeed(ObjectSeed):
#                                    NewData.pop(NewData.index(ObjectSeed))
#                           SeedCounter+=1
#                       print(str(InitialDataLength), "2-track vertices were merged into", str(len(NewData)), 'vertices with higher multiplicity...')
#
#                       open_file = open(output_file_location, "wb")
#                       pickle.dump(NewData, open_file)
#                       open_file.close()
#     del VertexPool
#     gc.collect()
#     status=4
#
#
#
# if status==4:
#     print(UF.TimeStamp(),'Current stage is no:', status)
#     VertexPool=[]
#     for i in range(0,int(round(float(args.DataCut)*0.6*0.6*0.6,0))):
#                   output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Merged_Seeds4_'+str(i)+'.pkl'
#                   data_file=open(output_file_location,'rb')
#                   NewData=pickle.load(data_file)
#                   data_file.close()
#                   VertexPool+=NewData
#     InitialDataLength=len(VertexPool)
#     SeedCounter=0
#     SeedCounterContinue=True
#     while SeedCounterContinue:
#         if SeedCounter==len(VertexPool):
#                           SeedCounterContinue=False
#                           break
#         progress=round((float(SeedCounter)/float(len(VertexPool)))*100,0)
#         print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
#         SubjectSeed=VertexPool[SeedCounter]
#         for ObjectSeed in VertexPool[SeedCounter+1:]:
#                     if SubjectSeed.InjectSeed(ObjectSeed):
#                                 VertexPool.pop(VertexPool.index(ObjectSeed))
#         SeedCounter+=1
#     print(str(InitialDataLength), "vertices from different files were merged into", str(len(VertexPool)), 'vertices with higher multiplicity...')
#     for v in range(0,len(VertexPool)):
#         VertexPool[v].AssignCNNVxId(v)
#     output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_REC_VERTICES.pkl'
#
#     print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
#     print(UF.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_file_location+bcolors.ENDC)
#     open_file = open(output_file_location, "wb")
#     pickle.dump(VertexPool, open_file)
#     open_file.close()
#     #UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R5', ['R5_R5'], "SoftUsed == \"EDER-VIANN-R5\"")
#     print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
#     print(UF.TimeStamp(),bcolors.OKGREEN+'The vertex merging has been completed..'+bcolors.ENDC)
#     print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
