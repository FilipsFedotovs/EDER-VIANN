#import libraries
import numpy as np
from numpy.random import randint
import argparse
import ast
import csv
import os
from os import path
import random as rnd
from random import random
import pandas as pd
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Set the parsing module
parser = argparse.ArgumentParser(description='Enter training job parameters')
parser.add_argument('--Mode',help="Please enter the running mode: 'R' for reset, 'C' for continuing the training", default='C')
parser.add_argument('--ModelName',help="Which model would you like to use as a base for training (please enter N if you want to train a new model from scratch)", default='N')
parser.add_argument('--ModelNewName',help="Would you like to save your pretrained model as a separate one", default='Default')
parser.add_argument('--LR',help="Would you like to modify the model Learning Rate, If yes please enter it here eg: 0.01 ", default='Default')
parser.add_argument('--EpochLength',help="How long do you want your epochs to be?", default='1000')
args = parser.parse_args()
#setting main learning parameters
mode=args.Mode
#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()

#Loading Data configurations
EOSsubDIR=EOS_DIR+'/'+'EDER-VIANN'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF
import Parameters as PM
if args.ModelName=='Default':
    ModelName=PM.CNN_Model_Name
else:
    ModelName=args.ModelName
epoch_len=int(args.EpochLength)
print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising EDER-VIANN model training module   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################            Written by Filips Fedotovs            #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################               PhD Student at UCL                 #########################"+bcolors.ENDC)
print(bcolors.HEADER+"###################### For troubleshooting please contact filips.fedotovs@cern.ch ##################"+bcolors.ENDC)
print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
#This code fragment covers the Algorithm logic on the first run
OptionHeader = [' --Mode ', ' --ImageSet ', ' --EOS ', " --AFS ", " --DNA ",
                 " --LR ", " --Epoch ", " --ModelName ", " --ModelNewName ", ' --EpochLength ']
if mode=='R' and args.ModelName=='N':
 log_name=EOSsubModelDIR+'/'+ args.ModelNewName + '.csv'
 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'GM5', ['GM5_GM5','GM5_PERFORMANCE_',args.ModelNewName], "SoftUsed == \"EDER-VIANN-GM5\"")
 job=[]
 job.append(1)
 job.append(1)
 job.append(PM.ModelArchitecture)
 job.append(args.LR)
 job.append(ModelName)

 DNA='"'+str(PM.ModelArchitecture)+'"'
 if args.ModelNewName=='Default':
     job.append(ModelName)
     OptionLine = ['Create', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1, ModelName, ModelName, epoch_len]
 else:
     job.append(args.ModelNewName)
     OptionLine = ['Create', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1, ModelName, args.ModelNewName, epoch_len]
 job.append(epoch_len)
 print(UF.TimeStamp(),bcolors.OKGREEN+'Job description has been created'+bcolors.ENDC)

 SHName = AFS_DIR + '/HTCondor/SH/SH_GM5.sh'
 SUBName = AFS_DIR + '/HTCondor/SUB/SUB_GM5.sub'
 MSGName = AFS_DIR + '/HTCondor/MSG/MSG_GM5'
 ScriptName = AFS_DIR + '/Code/Utilities/GM5_TrainModel_Sub.py '
 UF.SubmitJobs2Condor(
     [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-VIANN-GM5', False,
      False])
 job[4]=job[5]
 UF.LogOperations(EOSsubModelDIR+'/GM5_GM5_JobTask.csv','StartLog',[job])
 print(bcolors.BOLD+"Please the job completion in few hours by running this script with the option C"+bcolors.ENDC)
elif mode=='R':
 print(bcolors.FAIL+"Unfinished code, please run with --ModelName N for now"+bcolors.ENDC)
 exit()
 # UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'GM5', ['GM5_GM5','GM5_PERFORMANCE_'], "SoftUsed == \"EDER-VIANN-GM5\"")
 # job=[]
 # job.append(1)
 # job.append(1)
 # job.append(PM.ModelArchitecture)
 # job.append(args.LR)
 # job.append(ModelName)
 # DNA = '"' + str(PM.ModelArchitecture) + '"'
 # if args.ModelNewName=='Default':
 #     job.append(ModelName)
 #     OptionLine = ['Train', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1,  ModelName, ModelName]
 # else:
 #     job.append(args.ModelNewName)
 #     OptionLine = ['Train', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1, ModelName, args.ModelNewName]
 # print(UF.TimeStamp(),bcolors.OKGREEN+'Job description has been created'+bcolors.ENDC)
 # PerformanceHeader=[['Epochs','Set','Training Samples','Train Loss','Train Accuracy','Validation Loss','Validation Accuracy']]
 # UF.LogOperations(EOSsubModelDIR+'/GM5_PERFORMANCE_'+job[5]+'.csv','StartLog',PerformanceHeader)
 # OptionHeader = [' --Mode ', ' --ImageSet ', ' --EOS ', " --AFS ", " --DNA ",
 #                 " --LR ", " --Epoch ", " --ModelName ", " --ModelNewName "]
 # SHName = AFS_DIR + '/HTCondor/SH/SH_GM5.sh'
 # SUBName = AFS_DIR + '/HTCondor/SUB/SUB_GM5.sub'
 # MSGName = AFS_DIR + '/HTCondor/MSG/MSG_GM5'
 # ScriptName = AFS_DIR + '/Code/Utilities/GM5_TrainModel_Sub.py '
 # UF.SubmitJobs2Condor(
 #     [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-VIANN-GM5', True,
 #      True])
 # job[4]=job[5]
 # UF.LogOperations(EOSsubModelDIR+'/GM5_GM5_JobTask.csv','StartLog',[job])
 # print(bcolors.BOLD+"Please the job completion in few hours by running this script with the option C"+bcolors.ENDC)
if mode=='C':
   CurrentSet=0
   print(UF.TimeStamp(),'Continuing the training that has been started before')
   print(UF.TimeStamp(),'Checking the previous job completion...')
   csv_reader=open(EOSsubModelDIR+'/GM5_GM5_JobTask.csv',"r")
   PreviousJob = list(csv.reader(csv_reader))
   if args.LR!='Default':
       PreviousJob[0][3]=args.LR
   csv_reader.close()
   CurrentEpoch=int(PreviousJob[0][1])
   log_name=EOSsubModelDIR+'/'+ PreviousJob[0][5] + '.csv'
   try:
       Result=UF.LogOperations(log_name,'ReadLog', '_')
   except:
     print(UF.TimeStamp(),bcolors.WARNING+'Warning, the HTCondor job is still running'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to wait and try again later please enter W'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
     UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
     if UserAnswer=='W':
         print(UF.TimeStamp(),'OK, exiting now then')
         exit()
     if UserAnswer=='R':
        OptionLine = ['Create', PreviousJob[0][0], EOS_DIR, AFS_DIR, '"'+str(PreviousJob[0][2])+'"', PreviousJob[0][3], PreviousJob[0][1], PreviousJob[0][4], PreviousJob[0][5],PreviousJob[0][6]]
        SHName = AFS_DIR + '/HTCondor/SH/SH_GM5.sh'
        SUBName = AFS_DIR + '/HTCondor/SUB/SUB_GM5.sub'
        MSGName = AFS_DIR + '/HTCondor/MSG/MSG_GM5'
        ScriptName = AFS_DIR + '/Code/Utilities/GM5_TrainModel_Sub.py '
        UF.SubmitJobs2Condor(
            [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-VIANN-GM5', False,
             False])
        print(UF.TimeStamp(), bcolors.OKGREEN+"The Training Job has been resubmitted"+bcolors.ENDC)
        print(bcolors.OKGREEN+"Please check it in a few hours"+bcolors.ENDC)
        exit()

   CurrentEpoch=int(Result[-1][1])
   RequiredEpoch=int(PreviousJob[0][1])+int(PreviousJob[0][6])-1

   ###Working out the latest batch
   ###Working out the remaining jobs
   if CurrentEpoch!=RequiredEpoch:
     print(UF.TimeStamp(),bcolors.WARNING+'Warning, the HTCondor job is still running'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to wait and try again later please enter W'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
     UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
     if UserAnswer=='W':
         print(UF.TimeStamp(),'OK, exiting now then')
         exit()
     if UserAnswer=='R':
        OptionLine = ['Train', PreviousJob[0][0], EOS_DIR, AFS_DIR, '"'+str(PreviousJob[0][2])+'"', PreviousJob[0][3], PreviousJob[0][1], PreviousJob[0][4], PreviousJob[0][5],PreviousJob[0][6]]
        SHName = AFS_DIR + '/HTCondor/SH/SH_GM5.sh'
        SUBName = AFS_DIR + '/HTCondor/SUB/SUB_GM5.sub'
        MSGName = AFS_DIR + '/HTCondor/MSG/MSG_GM5'
        ScriptName = AFS_DIR + '/Code/Utilities/GM5_TrainModel_Sub.py '
        UF.SubmitJobs2Condor(
            [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-VIANN-GM5', True,
             True])
        print(UF.TimeStamp(), bcolors.OKGREEN+"The Training Job have been resubmitted"+bcolors.ENDC)
        print(bcolors.OKGREEN+"Please check it in a few hours"+bcolors.ENDC)
        exit()
   else:
      print(UF.TimeStamp(),bcolors.OKGREEN+'The training of the model',bcolors.OKBLUE+PreviousJob[0][4]+bcolors.ENDC, bcolors.OKGREEN+' by using the epochs ',str(PreviousJob[0][1]),'-',RequiredEpoch,'has been completed'+bcolors.ENDC)

      print(UF.TimeStamp(),'Training stats are below:')
      print(pd.DataFrame (Result[1:], columns = ['Set','Epoch', 'Training accuracy', 'testing accuracy', 'Train loss', 'Test loss']))
      print(bcolors.BOLD+'Would you like to continue training?'+bcolors.ENDC)
      UserAnswer=input(bcolors.BOLD+"Please, enter Y/N\n"+bcolors.ENDC)
      if UserAnswer=='Y':
            CurrentSet+=1
            PreviousJob[0][1]=str(int(PreviousJob[0][1])+int(PreviousJob[0][6]))
            UF.LogOperations(EOSsubModelDIR+'/GM5_GM5_JobTask.csv','StartLog',PreviousJob)
            OptionLine = ['Train', PreviousJob[0][0], EOS_DIR, AFS_DIR, '"'+str(PreviousJob[0][2])+'"', PreviousJob[0][3], PreviousJob[0][1], PreviousJob[0][4], PreviousJob[0][5],PreviousJob[0][6]]
            SHName = AFS_DIR + '/HTCondor/SH/SH_GM5.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_GM5.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_GM5'
            ScriptName = AFS_DIR + '/Code/Utilities/GM5_TrainModel_Sub.py '
            UF.SubmitJobs2Condor(
               [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-VIANN-GM5', False,
                False])
            print(UF.TimeStamp(),bcolors.OKGREEN+'The next epoch range ',str(int(PreviousJob[0][1])),'-',str(int(PreviousJob[0][1])+int(PreviousJob[0][6])-1),'has been submitted to HTCondor'+bcolors.ENDC)
            print(bcolors.BOLD,'Please run the script in few hours with --MODE C setting'+bcolors.ENDC)
            print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)

      if UserAnswer=='N':
            print(UF.TimeStamp(),bcolors.OKGREEN+'Training is finished then, thank you and good bye'+bcolors.ENDC)
            print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()


