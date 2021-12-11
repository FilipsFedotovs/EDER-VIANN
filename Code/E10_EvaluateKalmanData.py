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
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of MC truth to calculate reconstruction perfromance.')
parser.add_argument('--sf',help="Please choose the input file", default=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E9_KALMAN_REC_SEEDS.csv')
parser.add_argument('--of',help="Please choose the evaluation file (has to match the same geometrical domain and type of the track as the subject", default=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E3_TRUTH_SEEDS.csv')
parser.add_argument('--FullVxAnalysis',help="Should we do a full vertex analysis: Y/N?", default='N')
parser.add_argument('--sfv',help="Please choose the input track file", default=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E7_KALMAN_REC_VERTICES.csv')
parser.add_argument('--ofv',help="Please choose the input evaluation track file (has to match the same geometrical domain and type of the track as the subject", default=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E1_TRACKS.csv')
args = parser.parse_args()
MaxTracksPerJob = PM.MaxTracksPerJob
#Specifying the full path to input/output files
input_file_location=args.sf
input_eval_file_location=args.of
input_file_vx_location=args.sfv
input_eval_vx_location=args.ofv
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-VIANN Evaluation module             ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Analysing evaluation data... ',bcolors.ENDC)
if os.path.isfile(input_eval_file_location)!=True:
                 print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",input_eval_file_location,'is missing, please restart the evaluation sequence scripts'+bcolors.ENDC)
eval_data=pd.read_csv(input_eval_file_location,header=0,usecols=['Track_1','Track_2'])
eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Track_1'], eval_data['Track_2'])]
eval_data.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
eval_data.drop(eval_data.index[eval_data['Track_1'] == eval_data['Track_2']], inplace = True)
eval_data.drop(["Track_1"],axis=1,inplace=True)
eval_data.drop(["Track_2"],axis=1,inplace=True)
TotalMCVertices=len(eval_data.axes[0])
TotalRecVertices=0
MatchedVertices=0
FakeVertices=0
if os.path.isfile(input_file_location)!=True:
                 print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",input_file_location,'is missing, please restart the reconstruction sequence scripts'+bcolors.ENDC)
test_data=pd.read_csv(input_file_location,header=0,usecols=['Track_1','Track_2'])
test_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(test_data['Track_1'], test_data['Track_2'])]
test_data.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
test_data.drop(test_data.index[test_data['Track_1'] == test_data['Track_2']], inplace = True)
test_data.drop(["Track_1"],axis=1,inplace=True)
test_data.drop(["Track_2"],axis=1,inplace=True)
CurrentRecVertices=len(test_data.axes[0])
TotalRecVertices=CurrentRecVertices
test_data=pd.merge(test_data, eval_data, how="inner", on=["Seed_ID"])
RemainingRecVertices=len(test_data.axes[0])
MatchedVertices=RemainingRecVertices
FakeVertices=(CurrentRecVertices-RemainingRecVertices)
Recall=round((float(MatchedVertices)/float(TotalMCVertices))*100,2)
Precision=round((float(MatchedVertices)/float(TotalRecVertices))*100,2)
F1_Score=round(2*((Recall*Precision)/(Recall+Precision)),2)
print(UF.TimeStamp(), bcolors.OKGREEN+'FEDRA evaluation has been finished'+bcolors.ENDC)
print(bcolors.HEADER+"#########################################  Seed Level Evaluation Results  #########################################"+bcolors.ENDC)
print('Total 2-track combinations are expected according to Monte Carlo:',TotalMCVertices)
print('Total 2-track combinations were reconstructed by FEDRA:',TotalRecVertices)
print('FEDRA correct combinations were reconstructed:',MatchedVertices)
print('Therefore the recall of the current model is',bcolors.BOLD+str(Recall), '%'+bcolors.ENDC)
print('And the precision of the current model is',bcolors.BOLD+str(Precision), '%'+bcolors.ENDC)
print('The F1 score of the current model is',bcolors.BOLD+str(F1_Score), '%'+bcolors.ENDC)
if args.FullVxAnalysis=='Y':
    print(UF.TimeStamp(), bcolors.OKGREEN+'Evaluating Vertex level reconstruction performance'+bcolors.ENDC)
    print(UF.TimeStamp(),'Analysing evaluation and FEDRA full vertexing data... ',bcolors.ENDC)
    eval_data=pd.read_csv(input_eval_vx_location,header=0,usecols=['Track_ID','Mother_ID'])
    eval_data=eval_data.drop_duplicates()
    eval_data_vx_count=eval_data.groupby(by=['Mother_ID'])['Track_ID'].count().reset_index()
    eval_data_vx_count=eval_data_vx_count.rename(columns={'Track_ID': "MC_Multiplicity"})
    eval_data_vx_count=eval_data_vx_count.drop(eval_data_vx_count.index[eval_data_vx_count['MC_Multiplicity'] < 2])
    TotalFullVx=len(eval_data_vx_count)
    eval_data=pd.merge(eval_data, eval_data_vx_count, on=["Mother_ID"])
    fedra_data=pd.read_csv(input_file_vx_location,header=0,usecols=['Track_ID','Mother_ID'])
    fedra_data=fedra_data.drop_duplicates()
    fedra_data_vx_count=fedra_data.groupby(by=['Mother_ID'])['Track_ID'].count().reset_index()
    fedra_data_vx_count=fedra_data_vx_count.rename(columns={'Track_ID': "FEDRA_Multiplicity"})
    fedra_data_vx_count=fedra_data_vx_count.drop(fedra_data_vx_count.index[fedra_data_vx_count['FEDRA_Multiplicity'] < 2])
    TotalFullFedraVx=len(fedra_data_vx_count)
    fedra_data=pd.merge(fedra_data, fedra_data_vx_count, on=["Mother_ID"])
    fedra_data=fedra_data.rename(columns={'Mother_ID': "Vertex_ID"})
    combo_set=pd.merge(fedra_data, eval_data, on=["Track_ID"])
    fedra_track_map=combo_set.groupby(by=['Mother_ID','Vertex_ID','FEDRA_Multiplicity',"MC_Multiplicity"])['Track_ID'].count().reset_index()
    fedra_track_map=fedra_track_map.sort_values(['Vertex_ID','Mother_ID','Track_ID'],ascending=[1,1,0])
    fedra_track_map=fedra_track_map.rename(columns={'Track_ID': "Overlap"})
    fedra_track_map.drop_duplicates(subset="Vertex_ID",keep='first',inplace=True)
    fedra_track_map=fedra_track_map.drop(fedra_track_map.index[fedra_track_map['Overlap'] < 2])
    fedra_track_map['row_recall']=fedra_track_map['Overlap']/fedra_track_map['MC_Multiplicity']
    fedra_track_map['row_precision']=fedra_track_map['Overlap']/fedra_track_map['FEDRA_Multiplicity']
    overall_recall=round((len(fedra_track_map)/TotalFullVx)*100,2)
    overall_precision=round((len(fedra_track_map)/TotalFullFedraVx)*100,2)
    recall_table=fedra_track_map.groupby(by=["MC_Multiplicity"])['row_recall'].mean().reset_index()
    precision_table=fedra_track_map.groupby(by=["MC_Multiplicity"])['row_precision'].mean().reset_index()
    count_table=fedra_track_map.groupby(by=["MC_Multiplicity"])['row_precision'].count().reset_index()
    count_table=count_table.rename(columns={'row_precision': "No_Vx"})
    table=pd.merge(count_table, recall_table, how="inner", on=["MC_Multiplicity"])
    table=pd.merge(table, precision_table, how="inner", on=["MC_Multiplicity"])
    print('Multi-track vertices expected according to Monte Carlo:',TotalFullVx)
    print('Multi-track vertices reconstructed by FEDRA:',TotalFullFedraVx)
    print('Multi-track vertices that were at least partially correctly reconstructed by FEDRA:',len(fedra_track_map))
    print('Therefore the overall recall of FEDRA vertex reconstruction is',bcolors.BOLD+str(overall_recall), '%'+bcolors.ENDC)
    print('And the overall precision of the FEDRA is',bcolors.BOLD+str(overall_precision), '%'+bcolors.ENDC)
    print('The average matched vertex reconstruction completeness is ',bcolors.BOLD+str(round(fedra_track_map['row_recall'].mean(),4)*100), '%'+bcolors.ENDC)
    print('And the average vertex purity is',bcolors.BOLD+str(round(fedra_track_map['row_precision'].mean(),4)*100), '%'+bcolors.ENDC)
    output_file_location = EOS_DIR + '/EDER-VIANN/Data/TEST_SET/E10_FEDRA_VX_REC_STATS.csv'
    table.to_csv(output_file_location,index=False)
    print(UF.TimeStamp(), bcolors.OKGREEN+"Stats have on FEDRA vertex reconstruction by Multiplicity has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



