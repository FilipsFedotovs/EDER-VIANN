#This simple script prepares the evaluation data for vertexing procedure

########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd
import ast

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
parser = argparse.ArgumentParser(description='This script prepares the MC tracking data for EDER-VIANN vertexing evaluation routines')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/user/a/aiuliano/public/sims_fedra/CH1_pot_03_02_20/b000001/b000001_withvertices.csv')
parser.add_argument('--Track',help="Which tracks to use FEDRA/MC (For actual vertexing the track designations used by FEDRA should be used", default='FEDRA')
parser.add_argument('--Xmin',help="This option restricts data to only those events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option restricts data to only those events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option restricts data to only those events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option restricts data to only those events that have tracks with hits y-coordinates that are below this value", default='0')
parser.add_argument('--RemoveTracksZ',help="This option enables to remove particular tracks of sarting Z-coordinate", default='[]')
########################################     Main body functions    #########################################
args = parser.parse_args()
input_file_location=args.f
RemoveTracksZ=ast.literal_eval(args.RemoveTracksZ)
Track=args.Track
Xmin=float(args.Xmin)
Xmax=float(args.Xmax)
Ymin=float(args.Ymin)
Ymax=float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0
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
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"####################  Initialising EDER-VIANN evaluation data preparation module     ###################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules have been imported successfully..."+bcolors.ENDC)
#fetching_test_data
print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
if Track=='FEDRA':

 data=pd.read_csv(input_file_location,
            header=0,
            usecols=[PM.FEDRA_Track_ID,PM.FEDRA_Track_QUADRANT,PM.x,PM.y,PM.z,PM.MC_VX_ID,PM.MC_Event_ID, PM.MC_VX_PDG])

 total_rows=len(data.axes[0])
 print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
 print(UF.TimeStamp(),'Removing unreconstructed hits...')
 data[PM.MC_VX_ID] = data[PM.MC_VX_ID].astype(str)
 data[PM.MC_VX_PDG] = data[PM.MC_VX_PDG].astype(str)
 print(UF.TimeStamp(),'Removing invalid vertices...')
 for val in PM.MC_NV_VX_ID:
    data.drop(data.index[data[PM.MC_VX_ID] == str(val)], inplace = True)
 print(UF.TimeStamp(),'Keeping only signal vertices...')
 for val in PM.MC_SGNL_VX_PDG:
    data.drop(data.index[data[PM.MC_VX_PDG] != str(val)], inplace = True)
 data=data.dropna()
 final_rows=len(data.axes[0])
 print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
 data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
 data[PM.FEDRA_Track_ID] = data[PM.FEDRA_Track_ID].astype(int)
 data[PM.FEDRA_Track_ID] = data[PM.FEDRA_Track_ID].astype(str)
 #data[PM.FEDRA_Track_QUADRANT] = data[PM.FEDRA_Track_QUADRANT].astype(int)
 data[PM.FEDRA_Track_QUADRANT] = data[PM.FEDRA_Track_QUADRANT].astype(str)
 data['Track_ID'] = data[PM.FEDRA_Track_QUADRANT] + '-' + data[PM.FEDRA_Track_ID]
 data['Mother_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_VX_ID]
 data['Mother_PDG'] = data[PM.MC_VX_PDG]
 data=data.drop([PM.FEDRA_Track_ID],axis=1)
 data=data.drop([PM.FEDRA_Track_QUADRANT],axis=1)
 data=data.drop([PM.MC_Event_ID],axis=1)
 data=data.drop([PM.MC_VX_ID],axis=1)
 compress_data=data.drop([PM.x,PM.y,PM.z,'Mother PDG'],axis=1)
 compress_data['Mother_No']= compress_data['Mother_ID']
 compress_data=compress_data.groupby(by=['Track_ID','Mother_ID',])['Mother_No'].count().reset_index()
 compress_data=compress_data.sort_values(['Track_ID','Mother_No'],ascending=[1,0])
 compress_data.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
 data=data.drop(['Mother_ID'],axis=1)
 compress_data=compress_data.drop(['Mother_No'],axis=1)
 data=pd.merge(data, compress_data, how="left", on=["Track_ID"])
 print(data)
 exit()
 if SliceData:
     print(UF.TimeStamp(),'Slicing the data...')
     ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
     ValidEvents.drop([PM.x,PM.y,PM.z,'Mother_ID','Mother_PDG'],axis=1,inplace=True)
     ValidEvents.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
     data=pd.merge(data, ValidEvents, how="inner", on=['Track_ID'])
     final_rows=len(data.axes[0])
     print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
 output_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E1_TRACKS.csv'

elif Track=='MC':
 data=pd.read_csv(input_file_location,
            header=0,
            usecols=[PM.MC_Event_ID,PM.MC_Track_ID,PM.x, PM.y,PM.z,PM.MC_VX_ID])
 total_rows=len(data.axes[0])
 print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
 print(UF.TimeStamp(),'Removing unreconstructed hits...')
 data=data.dropna()
 data[PM.MC_VX_ID] = data[PM.MC_VX_ID].astype(str)
 data.drop(data.index[data[PM.MC_VX_ID] == PM.MC_NV_VX_ID], inplace = True)
 final_rows=len(data.axes[0])
 print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
 data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(int)
 data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
 data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(int)
 data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
 data['Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID]
 data['Mother_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_VX_ID]
 data=data.drop([PM.MC_Track_ID],axis=1)
 data=data.drop([PM.MC_VX_ID],axis=1)
 if SliceData:
     print(UF.TimeStamp(),'Slicing the data...')
     ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
     ValidEvents.drop([PM.x,PM.y,PM.z,'Track_ID','Mother_ID','Mother_PDG'],axis=1,inplace=True)
     ValidEvents.drop_duplicates(subset=PM.MC_Event_ID,keep='first',inplace=True)
     data=pd.merge(data, ValidEvents, how="inner", on=[PM.MC_Event_ID])
     final_rows=len(data.axes[0])
     print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
 data=data.drop([PM.MC_Event_ID],axis=1)
 output_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E1_TRACKS.csv'
else:
  print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
  print(UF.TimeStamp(), bcolors.FAIL+"No valid option for track reconstruction source has been chosen (FEDRA/MC), aborting the script..."+bcolors.ENDC)
  exit()
print(UF.TimeStamp(),'Removing tracks which have less than',PM.MinHitsTrack,'hits...')
track_no_data=data.groupby(['Track_ID'],as_index=False).count()
track_no_data=track_no_data.drop([PM.y,PM.z,'Mother_ID','Mother_PDG'],axis=1)
track_no_data=track_no_data.rename(columns={PM.x: "Track_No"})
new_combined_data=pd.merge(data, track_no_data, how="left", on=["Track_ID"])
new_combined_data = new_combined_data[new_combined_data.Track_No >= PM.MinHitsTrack]
new_combined_data = new_combined_data.drop(['Track_No'],axis=1)
new_combined_data=new_combined_data.sort_values(['Track_ID',PM.x],ascending=[1,1])
grand_final_rows=len(new_combined_data.axes[0])
print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
if len(RemoveTracksZ)>1:
    print(UF.TimeStamp(),'Removing tracks based on start point')
    TracksZdf = pd.DataFrame(RemoveTracksZ, columns = ['Bad_z'], dtype=float)
    new_combined_data_aggregated=new_combined_data.groupby(['Track_ID'])['z'].min().reset_index()
    new_combined_data_aggregated=new_combined_data_aggregated.rename(columns={'z': "PosBad_Z"})
    new_combined_data=pd.merge(new_combined_data, new_combined_data_aggregated, how="left", on=["Track_ID"])
    new_combined_data=pd.merge(new_combined_data, TracksZdf, how="left", left_on=["PosBad_Z"], right_on=['Bad_z'])
    new_combined_data=new_combined_data[new_combined_data['Bad_z'].isnull()]
    new_combined_data=new_combined_data.drop(columns=['Bad_z', 'PosBad_Z'])
print(UF.TimeStamp(),'The cleaned data has ',len(new_combined_data),' hits')
new_combined_data.to_csv(output_file_location,index=False)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"The track data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()
