#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import pickle
import os
import math
import gc

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
parser = argparse.ArgumentParser(description='This script takes refined 2-track seed candidates from previous step and perfromes a vertex fit by using pre-trained CNN model.')
parser.add_argument('--Mode',help="Running Mode: Reset(R)/Continue(C)", default='C')

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
from Utility_Functions import Seed
import Parameters as PM #This is where we keep framework global parameters
########################################     Preset framework parameters    #########################################
MaxSeedsPerJob = PM.MaxSeedsPerVxPool
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R4_R5_CNN_Fit_Seeds.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-VIANN Seed Link Analysis module    ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,names = ['Track_1','Track_2'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
No_Fractions=int(math.ceil(len(data)/MaxSeedsPerJob))
del data
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to vertex the seeds from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed Analysis jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R5', ['R5_R5_'], "SoftUsed == \"EDER-VIANN-R5\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      OptionHeader = [' --Fraction ', ' --EOS ', " --AFS ", " --MaxSeedsPerJob "]
      OptionLine = ['$1', EOS_DIR, AFS_DIR, MaxSeedsPerJob]
      SHName = AFS_DIR + '/HTCondor/SH/SH_R5.sh'
      SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R5.sub'
      MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R5'
      ScriptName = AFS_DIR + '/Code/Utilities/R5_AnalyseSeedLinks_Sub.py '
      UF.SubmitJobs2Condor(
          [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, No_Fractions, 'EDER-VIANN-R5', False,
           False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   print(UF.TimeStamp(),'Checking results... ',bcolors.ENDC)
   bad_pop = []
   for f in range(No_Fractions):
       req_file=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Link_CNN_Fit_Seeds_'+str(f)+'.csv'
       if os.path.isfile(req_file) == False:
           job_details = [f, MaxSeedsPerJob, AFS_DIR, EOS_DIR]
           OptionHeader = [' --Fraction ', ' --EOS ', " --AFS ", " --MaxSeedsPerJob "]
           OptionLine = [f, EOS_DIR, AFS_DIR, MaxSeedsPerJob]
           SHName = AFS_DIR + '/HTCondor/SH/SH_R5_'+str(f)+'.sh'
           SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R5_'+str(f)+'.sub'
           MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R5_'+str(f)
           ScriptName = AFS_DIR + '/Code/Utilities/R5_AnalyseSeedLinks_Sub.py '
           job_details=[OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-VIANN-R5', False,
           False]
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
        exit()
        for bp in bad_pop:
             UF.SubmitJobs2Condor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Seed Creation jobs have finished'+bcolors.ENDC)
       print(UF.TimeStamp(),'Collating the results...')
       for f in range(No_Fractions):
           req_file = EOS_DIR + '/EDER-VIANN/Data/REC_SET/R5_R5_Link_CNN_Fit_Seeds_' + str(f) + '.csv'
           progress=round((float(f)/float(No_Fractions))*100,2)
           print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
           if os.path.isfile(req_file)==False:
                 print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",req_file,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
           elif os.path.isfile(req_file):
                 if (f)==0:
                     base_data = pd.read_csv(req_file,usecols=['Track_1', 'Track_2','Seed_CNN_Fit','Link_Strength','AntiLink_Strenth'])
                 else:
                     new_data = pd.read_csv(req_file,usecols=['Track_1', 'Track_2','Seed_CNN_Fit','Link_Strength','AntiLink_Strenth'])
                     frames = [base_data, new_data]
                     base_data = pd.concat(frames,ignore_index=True)
       Records=len(base_data)
       print(UF.TimeStamp(),'The pre-analysed reconstructed set contains', Records, '2-track link-fitted seeds',bcolors.ENDC)
       base_data['Seed_Link_Fit'] = base_data.apply(PM.Seed_Bond_Fit_Acceptance,axis=1)
       base_data['Seed_Index'] = base_data.index
       base_data.drop(base_data.index[base_data['Seed_Link_Fit'] < PM.link_acceptance],inplace=True)  # Dropping the seeds that don't pass the link fit threshold
       base_data.drop(base_data.index[base_data['Seed_CNN_Fit'] < PM.pre_vx_acceptance],inplace=True)  # Dropping the seeds that don't pass the link fit threshold
       Records_After_Compression=len(base_data)
       output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_E4_LINK_FIT_SEEDS.csv'
       print(UF.TimeStamp(),'Out of', Records, ' seeds, only', Records_After_Compression, 'pass the link selection criteria...' ,bcolors.ENDC)
       base_data.to_csv(output_file_location,index=False)
       object_file_location = EOS_DIR + '/EDER-VIANN/Data/REC_SET/R4_R5_CNN_Fit_Seeds.pkl'
       print(UF.TimeStamp(), bcolors.OKGREEN + "Evaluation result is saved in" + bcolors.ENDC,bcolors.OKBLUE + output_file_location + bcolors.ENDC)
       print(UF.TimeStamp(), 'Decorating seed objects in ' + bcolors.ENDC,bcolors.OKBLUE + object_file_location + bcolors.ENDC)
       base_data=base_data.values.tolist()
       new_data=[]
       for b in base_data:
           new_data.append(b[6])
       base_data=new_data
       del new_data
       print(UF.TimeStamp(), 'Loading seed object data from ', bcolors.OKBLUE + object_file_location + bcolors.ENDC)
       object_file = open(object_file_location, 'rb')
       object_data = pickle.load(object_file)
       object_file.close()
       selected_objects=[]
       for nd in range(len(base_data)):
           selected_objects.append(object_data[base_data[nd]])
           progress = round((float(nd) / float(len(base_data))) * 100, 1)
           print(UF.TimeStamp(), 'Refining the seed objects, progress is ', progress, ' %', end="\r", flush=True)  # Progress display
       del object_data
       del base_data
       gc.collect()
       object_file_location = EOS_DIR + '/EDER-VIANN/Data/REC_SET/R5_R6_Link_Fit_Seeds.pkl'
       obj_data_file=open(object_file_location,'wb')
       pickle.dump(selected_objects,obj_data_file)
       obj_data_file.close()
       print(UF.TimeStamp(), bcolors.OKGREEN + str(len(selected_objects))+" seed objects are saved in" + bcolors.ENDC,bcolors.OKBLUE + object_file_location + bcolors.ENDC)
       print(bcolors.BOLD+'Would you like to clean up?'+bcolors.ENDC)
       UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
       if UserAnswer=='Y':
            UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R4', ['R4_R5'], "SoftUsed == \"EDER-VIANN-R4\"")
            UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R5', ['R5_R5'], "SoftUsed == \"EDER-VIANN-R5\"")
            print(
                        bcolors.HEADER + "########################################################################################################" + bcolors.ENDC)
       else:
            print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)


