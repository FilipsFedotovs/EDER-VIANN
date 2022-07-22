#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
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
parser = argparse.ArgumentParser(description='This script takes the output from the previous step and decorates the mwith track hit information that can be used to render the seed image. This script creates teraining and validation samples.')
parser.add_argument('--Mode',help="Running Mode: Reset(R)/Continue(C)", default='C')
parser.add_argument('--Samples',help="How many samples? Please enter the number or ALL if you want to use all data", default='ALL')
parser.add_argument('--ValidationSize',help="What is the proportion of Validation Images?", default='0.1')
parser.add_argument('--LabelMix',help="What is the desired proportion of genuine vertices in the training/validation sets", default='0.5')

######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode




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
import Parameters as PM #This is where we keep framework global parameters
########################################     Preset framework parameters    #########################################
VO_max_Z=PM.VO_max_Z
VO_min_Z=PM.VO_min_Z
VO_T=PM.VO_T
MaxDoca=PM.MaxDoca
MinAngle=PM.MinAngle
MaxAngle=PM.MaxAngle
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
MaxTracksPerJob = PM.MaxTracksPerJob
MaxSeedsPerJob = PM.MaxSeedsPerJob
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM1_TRACKS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-VIANN Image Generation module      ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['Track_ID','z'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
data = data.groupby('Track_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
data=data.reset_index()
data = data.groupby('z')['Track_ID'].count()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
data=data.reset_index()
data=data.sort_values(['z'],ascending=True)
data['Sub_Sets']=np.ceil(data['Track_ID']/MaxTracksPerJob)
data['Sub_Sets'] = data['Sub_Sets'].astype(int)
data = data.values.tolist() #Convirting the result to List data type
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to create the seeds from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed Creation jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'GM3', ['GM3_GM3','GM3_VALIDATION','GM3_TRAIN'], "SoftUsed == \"EDER-VIANN-GM3\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for j in range(0,len(data)):
        for sj in range(0,int(data[j][2])):
            f_count=0
            for f in range(0,1000):
             new_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM2_GM3_RawSeeds_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
             if os.path.isfile(new_output_file_location):
                 f_count=f
            OptionHeader = [' --Set ', ' --SubSet ', ' --Fraction ', ' --EOS ', " --AFS ", " --VO_T ", " --VO_max_Z ",
                            " --VO_min_Z ", " --MaxDoca ", " --MinAngle ", " --MaxAngle "]
            OptionLine = [j, sj, '$1', EOS_DIR, AFS_DIR, VO_T, VO_max_Z, VO_min_Z, MaxDoca, MinAngle, MaxAngle]
            SHName = AFS_DIR + '/HTCondor/SH/SH_GM3_' + str(j) + '_' + str(sj) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_GM3_' + str(j) + '_' + str(sj) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_GM3_' + str(j) + '_' + str(sj)
            ScriptName = AFS_DIR + '/Code/Utilities/GM3_GenerateImages_Sub.py '
            UF.SubmitJobs2Condor(
                [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, f_count + 1, 'EDER-VIANN-GM3', False,
                 False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   ProcessStatus=1
   bad_pop=[]
   MaxJ=0
   for j in range(0,len(data)):
       progress=int( round( (float(j)/float(len(data))*100),0)  )
       print(UF.TimeStamp(),"Checking jobs, progress is ",progress,' %',end="\r", flush=True)
       for sj in range(0,int(data[j][2])):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM2_GM3_RawSeeds_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
              required_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_RawImages_'+str(j)+'_'+str(sj)+'_'+str(f)+'.pkl'
              OptionHeader = [' --Set ', ' --SubSet ', ' --Fraction ', ' --EOS ', " --AFS ", " --VO_T ", " --VO_max_Z ",
                              " --VO_min_Z ", " --MaxDoca ", " --MinAngle ", " --MaxAngle "]
              OptionLine = [j, sj, f, EOS_DIR, AFS_DIR, VO_T, VO_max_Z, VO_min_Z, MaxDoca, MinAngle,
                            MaxAngle]
              SHName = AFS_DIR + '/HTCondor/SH/SH_GM3_' + str(j) + '_' + str(sj) + '_' + str(f) + '.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_GM3_' + str(j) + '_' + str(sj) + '_' + str(f) + '.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_GM3_' + str(j) + '_' + str(sj) + '_' + str(f)
              ScriptName = AFS_DIR + '/Code/Utilities/GM3_GenerateImages_Sub.py '
              job_details = [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-VIANN-GM3', False,
                             False]
              if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                 bad_pop.append(job_details)
              if os.path.isfile(required_output_file_location):
                  MaxJ=j
   if len(bad_pop)>0:
     print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to wait and try again later please enter W'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
     UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
     if UserAnswer=='W':
         print(UF.TimeStamp(),'OK, exiting now then')
         exit()
     if UserAnswer=='R':
        for bp in bad_pop:
             UF.SubmitJobs2Condor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:
       test_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_CondensedImages_'+str(MaxJ)+'.pkl'
       if os.path.isfile(test_file):
           print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
           print(UF.TimeStamp(), bcolors.OKGREEN+"The process has been ran before, continuing the image generation"+bcolors.ENDC)
           ProcessStatus=2
       test_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_SamplesCondensedImages_'+str(MaxJ)+'.pkl'
       if os.path.isfile(test_file):
           print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
           print(UF.TimeStamp(), bcolors.OKGREEN+"The process has been ran before and image sampling has begun"+bcolors.ENDC)
           ProcessStatus=3
       if ProcessStatus==1:
           UF.LogOperations(EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_Temp_Stats.csv','StartLog', [[0,0]])
           print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Seed Creation jobs have finished'+bcolors.ENDC)
           print(UF.TimeStamp(),'Collating the results...')
           for j in range(0,len(data)):
             output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_CondensedImages_'+str(j)+'.pkl'
             if os.path.isfile(output_file_location)==False:
                Temp_Stats=UF.LogOperations(EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_Temp_Stats.csv','ReadLog', '_')
                TotalImages=int(Temp_Stats[0][0])
                TrueSeeds=int(Temp_Stats[0][1])
                for sj in range(0,int(data[j][2])):
                   for f in range(0,1000):
                      new_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM2_GM3_RawSeeds_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
                      required_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_RawImages_'+str(j)+'_'+str(sj)+'_'+str(f)+'.pkl'
                      if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                         print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",required_output_file_location,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
                      elif os.path.isfile(required_output_file_location):
                         if (sj)==(f)==0:
                            base_data_file=open(required_output_file_location,'rb')
                            base_data=pickle.load(base_data_file)
                            base_data_file.close()
                         else:
                            new_data_file=open(required_output_file_location,'rb')
                            new_data=pickle.load(new_data_file)
                            new_data_file.close()
                            base_data+=new_data
                try:
                 Records=len(base_data)
                 print(UF.TimeStamp(),'Set',str(j),'contains', Records, 'raw images',bcolors.ENDC)

                 base_data=list(set(base_data))
                 Records_After_Compression=len(base_data)
                 if Records>0:
                      Compression_Ratio=int((Records_After_Compression/Records)*100)
                 else:
                      CompressionRatio=0
                 TotalImages+=Records_After_Compression
                 TrueSeeds+=sum(1 for im in base_data if im.MC_truth_label == 1)
                 print(UF.TimeStamp(),'Set',str(j),'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                 open_file = open(output_file_location, "wb")
                 pickle.dump(base_data, open_file)
                 open_file.close()
                except:
                    continue
#                del new_data
                UF.LogOperations(EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_Temp_Stats.csv','StartLog', [[TotalImages,TrueSeeds]])
           ProcessStatus=2


       ####Stage 2
       if ProcessStatus==2:
           print(UF.TimeStamp(),'Sampling the required number of seeds',bcolors.ENDC)
           Temp_Stats=UF.LogOperations(EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_Temp_Stats.csv','ReadLog', '_')
           TotalImages=int(Temp_Stats[0][0])
           TrueSeeds=int(Temp_Stats[0][1])
           if args.Samples=='ALL':
               if TrueSeeds<=(float(args.LabelMix)*TotalImages):
                   RequiredTrueSeeds=TrueSeeds
                   RequiredFakeSeeds=int(round((RequiredTrueSeeds/float(args.LabelMix))-RequiredTrueSeeds,0))
               else:
                   RequiredFakeSeeds=TotalImages-TrueSeeds
                   RequiredTrueSeeds=int(round((RequiredFakeSeeds/(1.0-float(args.LabelMix)))-RequiredFakeSeeds,0))
           else:
               NormalisedTotSamples=int(args.Samples)
               if TrueSeeds<=(float(args.LabelMix)*NormalisedTotSamples):
                   RequiredTrueSeeds=TrueSeeds
                   RequiredFakeSeeds=int(round((RequiredTrueSeeds/float(args.LabelMix))-RequiredTrueSeeds,0))
               else:
                   RequiredFakeSeeds=NormalisedTotSamples*(1.0-float(args.LabelMix))
                   RequiredTrueSeeds=int(round((RequiredFakeSeeds/(1.0-float(args.LabelMix)))-RequiredFakeSeeds,0))
           if TrueSeeds==0:
               TrueSeedCorrection=0
           else:
              TrueSeedCorrection=RequiredTrueSeeds/TrueSeeds
           FakeSeedCorrection=RequiredFakeSeeds/(TotalImages-TrueSeeds)
           for j in range(0,len(data)):
              req_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_SamplesCondensedImages_'+str(j)+'.pkl'
              output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_CondensedImages_'+str(j)+'.pkl'
              if os.path.isfile(req_file)==False and os.path.isfile(output_file_location):
                  progress=int( round( (float(j)/float(len(data))*100),0)  )
                  print(UF.TimeStamp(),"Sampling image from the collated data, progress is ",progress,' % of seeds generated',end="\r", flush=True)
                  base_data_file=open(output_file_location,'rb')
                  base_data=pickle.load(base_data_file)
                  base_data_file.close()
                  ExtractedTruth=[im for im in base_data if im.MC_truth_label == 1]
                  ExtractedFake=[im for im in base_data if im.MC_truth_label == 0]
                  del base_data
                  gc.collect()
                  ExtractedTruth=random.sample(ExtractedTruth,int(round(TrueSeedCorrection*len(ExtractedTruth),0)))
                  ExtractedFake=random.sample(ExtractedFake,int(round(FakeSeedCorrection*len(ExtractedFake),0)))
                  TotalData=[]
                  TotalData=ExtractedTruth+ExtractedFake
                  write_data_file=open(req_file,'wb')
                  pickle.dump(TotalData, write_data_file)
                  write_data_file.close()
                  del TotalData
                  del ExtractedTruth
                  del ExtractedFake
                  gc.collect()
                  ProcessStatus=3


       if ProcessStatus==3:
           TotalData=[]
           for j in range(0,len(data)):
               output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM3_SamplesCondensedImages_'+str(j)+'.pkl'
               if os.path.isfile(output_file_location):
                  progress=int( round( (float(j)/float(len(data))*100),0)  )
                  print(UF.TimeStamp(),"Re-sampling image from the collated data, progress is ",progress,' % of seeds generated',end="\r", flush=True)
                  base_data_file=open(output_file_location,'rb')
                  base_data=pickle.load(base_data_file)
                  base_data_file.close()
                  TotalData+=base_data
           del base_data
           gc.collect()
           ValidationSampleSize=int(round(min((len(TotalData)*float(args.ValidationSize)),PM.MaxValSampleSize),0))
           random.shuffle(TotalData)
           output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM4_Validation_Set.pkl'
           ValExtracted_file = open(output_file_location, "wb")
           pickle.dump(TotalData[:ValidationSampleSize], ValExtracted_file)
           ValExtracted_file.close()
           TotalData=TotalData[ValidationSampleSize:]
           print(UF.TimeStamp(), bcolors.OKGREEN+"Validation Set has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
           No_Train_Files=int(math.ceil(len(TotalData)/PM.MaxTrainSampleSize))
           for SC in range(0,No_Train_Files):
             output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM3_GM4_Train_Set_'+str(SC+1)+'.pkl'
             OldExtracted_file = open(output_file_location, "wb")
             pickle.dump(TotalData[(SC*PM.MaxTrainSampleSize):min(len(TotalData),((SC+1)*PM.MaxTrainSampleSize))], OldExtracted_file)
             OldExtracted_file.close()
             print(UF.TimeStamp(), bcolors.OKGREEN+"Train Set", str(SC+1) ," has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
           UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'GM3', ['GM3_GM3_SamplesCondensedImages','GM3_GM3_CondensedImages'], "SoftUsed == \"EDER-VIANN-GM3\"")
           print(bcolors.BOLD+'Would you like to delete filtered seeds data?'+bcolors.ENDC)
           UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
           if UserAnswer=='Y':
               UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'GM3', ['GM2_GM3','GM3_GM3'], "SoftUsed == \"EDER-VIANN-GM3\"")
           else:
            print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
            print(UF.TimeStamp(), bcolors.OKGREEN+"Training and Validation data has been created: you can start working on the model..."+bcolors.ENDC)
            print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
            exit()
#End of the script



