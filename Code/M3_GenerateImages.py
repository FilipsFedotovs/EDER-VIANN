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
resolution=PM.resolution
acceptance=PM.acceptance
MaxX=PM.MaxX
MaxY=PM.MaxY
MaxZ=PM.MaxZ
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
MaxTracksPerJob = PM.MaxTracksPerJob
MaxSeedsPerJob = PM.MaxSeedsPerJob
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M1_TRACKS.csv'
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
      UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M3', ['M3_M3','M3_VALIDATION','M3_TRAIN'], "SoftUsed == \"EDER-VIANN-M3\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for j in range(0,len(data)):
        for sj in range(0,int(data[j][2])):
            f_count=0
            for f in range(0,1000):
             new_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M2_M3_RawSeeds_'+str(j+1)+'_'+str(sj+1)+'_'+str(f)+'.csv'
             if os.path.isfile(new_output_file_location):
                 f_count=f
            job_details=[(j+1),(sj+1),f_count,VO_max_Z,VO_min_Z,VO_T,MaxDoca,resolution,MaxX,MaxY,MaxZ,AFS_DIR,EOS_DIR]
            UF.BSubmitImageJobsCondor(job_details)
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   print(UF.TimeStamp(),'Checking results... ',bcolors.ENDC)
   test_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_VALIDATION_SET.csv'
   if os.path.isfile(test_file):
       print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(), bcolors.OKGREEN+"The process has been completed before, if you want to restart, please rerun with '--Mode R' option"+bcolors.ENDC)
       exit()
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for j in range(0,len(data)):
       for sj in range(0,int(data[j][2])):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M2_M3_RawSeeds_'+str(j+1)+'_'+str(sj+1)+'_'+str(f)+'.csv'
              required_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M3_Images_'+str(j+1)+'_'+str(sj+1)+'_'+str(f)+'.csv'
              job_details=[(j+1),(sj+1),f,VO_max_Z,VO_min_Z,VO_T,MaxDoca,resolution,MaxX,MaxY,MaxZ,AFS_DIR,EOS_DIR]
              if os.path.isfile(required_output_file_location)!=True  and os.path.isfile(new_output_file_location):
                 bad_pop.append(job_details)
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
             UF.SubmitImageJobsCondor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Seed Creation jobs have finished'+bcolors.ENDC)
       TotalImages=0
       TrueSeeds=0
       print(UF.TimeStamp(),'Collating the results...')
       for j in range(0,len(data)):
        for sj in range(0,int(data[j][2])):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M2_M3_RawSeeds_'+str(j+1)+'_'+str(sj+1)+'_'+str(f)+'.csv'
              required_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M3_Images_'+str(j+1)+'_'+str(sj+1)+'_'+str(f)+'.csv'
              if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                 print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",required_output_file_location,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
              elif os.path.isfile(required_output_file_location):
                 if (sj+1)==(f+1)==1:
                    base_data=pd.read_csv(required_output_file_location,names=['Seed_ID','Track','Label'])
                    base_data['Seed_ID_temp'] = base_data['Seed_ID']
                    base_data[['Seed_ID_1','Seed_ID_2']] = base_data['Seed_ID_temp'].str.split(',',expand=True)
                    base_data['Seed_ID_1'] =  base_data['Seed_ID_1'].apply(lambda x: x.replace('[','').replace(']',''))
                    base_data['Seed_ID_2'] =  base_data['Seed_ID_2'].apply(lambda x: x.replace('[','').replace(']',''))
                    base_data["Seed_ID_temp"]= ['-'.join(sorted(tup)) for tup in zip(base_data['Seed_ID_1'], base_data['Seed_ID_2'])]
                    base_data.drop(base_data.index[base_data['Seed_ID_1'] == base_data['Seed_ID_2']], inplace = True)
                    base_data.drop(['Seed_ID_1'],axis=1,inplace=True)
                    base_data.drop(['Seed_ID_2'],axis=1,inplace=True)
                 else:
                    new_data=pd.read_csv(required_output_file_location,names=['Seed_ID','Track','Label'])
                    new_data['Seed_ID_temp'] = new_data['Seed_ID']
                    new_data[['Seed_ID_1','Seed_ID_2']] = new_data['Seed_ID_temp'].str.split(',',expand=True)
                    new_data['Seed_ID_1'] =  new_data['Seed_ID_1'].apply(lambda x: x.replace('[','').replace(']',''))
                    new_data['Seed_ID_2'] =  new_data['Seed_ID_2'].apply(lambda x: x.replace('[','').replace(']',''))
                    new_data["Seed_ID_temp"]= ['-'.join(sorted(tup)) for tup in zip(new_data['Seed_ID_1'], new_data['Seed_ID_2'])]
                    new_data.drop(new_data.index[new_data['Seed_ID_1'] == new_data['Seed_ID_2']], inplace = True)
                    new_data.drop(['Seed_ID_1'],axis=1,inplace=True)
                    new_data.drop(['Seed_ID_2'],axis=1,inplace=True)
                    frames=[base_data,new_data]
                    base_data=pd.concat(frames)

        try:
         Records=len(base_data.axes[0])
         print(UF.TimeStamp(),'Set',str(j+1),'contains', Records, 'images',bcolors.ENDC)
         output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M3_CondensedImages_'+str(j+1)+'.csv'
         base_data.drop_duplicates(subset="Seed_ID_temp",keep='first',inplace=True)
         base_data.drop(["Seed_ID_temp"],axis=1,inplace=True)
         TrueSeeds+=base_data.groupby(['Label']).size()['[1]']
         TotalImages+=Records
         Records_After_Compression=len(base_data.axes[0])
         if Records>0:
              Compression_Ratio=int((Records_After_Compression/Records)*100)
         else:
              CompressionRatio=0
         print(UF.TimeStamp(),'Set',str(j+1),'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
         base_data.to_csv(output_file_location,index=False)
        except:
            continue
       del new_data
       print(UF.TimeStamp(),'Sampling the required number of seeds',bcolors.ENDC)
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
       TrueSeedCorrection=RequiredTrueSeeds/TrueSeeds
       FakeSeedCorrection=RequiredFakeSeeds/(TotalImages-TrueSeeds)
       for j in range(0,len(data)):
          progress=int( round( (float(j)/float(len(data))*100),0)  )
          print(UF.TimeStamp(),"Sampling image from the collated data, progress is ",progress,' % of seeds generated')
          try:
           output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M3_CondensedImages_'+str(j+1)+'.csv'
           base_data=pd.read_csv(output_file_location,names=['Seed_ID','Track','Label'])
           CurrentTrueSeeds=base_data.groupby(['Label']).size()['[1]']
           CurrentFakeSeeds=base_data.groupby(['Label']).size()['[0]']
           CurrentTrueSeedCorrection=CurrentTrueSeeds/TrueSeeds
           CurrentFakeSeedCorrection=CurrentFakeSeeds/(TotalImages-TrueSeeds)
           if j==0:
              OldExtractedTruth=base_data[base_data['Label']=='[1]']
              OldExtractedFake=base_data[base_data['Label']=='[0]']
              OldExtractedTruth=OldExtractedTruth.sample(frac=(TrueSeedCorrection), random_state=1)
              OldExtractedFake=OldExtractedFake.sample(frac=(FakeSeedCorrection), random_state=1)
           else:
             NewExtractedTruth=base_data[base_data['Label']=='[1]']
             NewExtractedFake=base_data[base_data['Label']=='[0]']
             NewExtractedTruth=NewExtractedTruth.sample(frac=(TrueSeedCorrection), random_state=1)
             NewExtractedFake=NewExtractedFake.sample(frac=(FakeSeedCorrection), random_state=1)
             true_frames=[OldExtractedTruth,NewExtractedTruth]
             OldExtractedTruth=pd.concat(true_frames)
             fake_frames=[OldExtractedFake,NewExtractedFake]
             OldExtractedTruth=pd.concat(true_frames)
             OldExtractedFake=pd.concat(fake_frames)
          except:
              continue
       combined_frames=[OldExtractedFake,OldExtractedTruth]

       OldExtracted=pd.concat(combined_frames)
       del OldExtractedFake
       del OldExtractedTruth
       del NewExtractedTruth
       del NewExtractedFake
       del base_data
       gc.collect()
       ValidationSampleSize=int(round(min((len(OldExtracted.axes[0])*float(args.ValidationSize)),PM.MaxValSampleSize),0))
       OldExtracted = OldExtracted.sample(frac=1).reset_index(drop=True)
       output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_VALIDATION_SET.csv'
       ValExtracted = OldExtracted.sample(n=ValidationSampleSize, random_state=1)
       ValExtracted.to_csv(output_file_location,index=False,header=False)
       print(UF.TimeStamp(), bcolors.OKGREEN+"Validation Set has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
       OldExtracted=OldExtracted.drop(ValExtracted.index)
       No_Train_Files=int(math.ceil(len(OldExtracted.axes[0])/PM.MaxTrainSampleSize))
       for SC in range(0,No_Train_Files):
         output_file_location=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_TRAIN_SET_'+str(SC+1)+'.csv'
         OldExtracted[(SC*PM.MaxTrainSampleSize):min(len(OldExtracted.axes[0]),((SC+1)*PM.MaxTrainSampleSize))].to_csv(output_file_location,index=False,header=False)
         print(UF.TimeStamp(), bcolors.OKGREEN+"Train Set", str(SC+1) ," has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
       UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M3', ['M3_M3'], "SoftUsed == \"EDER-VIANN-M3\"")
       print(bcolors.BOLD+'Would you like to delete filtered seeds data?'+bcolors.ENDC)
       UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
       if UserAnswer=='Y':
           UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M3', ['M2_M3'], "SoftUsed == \"EDER-VIANN-M3\"")
       else:
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(), bcolors.OKGREEN+"Training and Validation data has been created: you can start working on the model..."+bcolors.ENDC)
        print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
        exit()
#End of the script



