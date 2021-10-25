#This simple script prepares data for CNN
########################################    Import libraries    #############################################
import Utility_Functions as UF
from Utility_Functions import Seed
import argparse
import pandas as pd #We use Panda for a routine data processing
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
parser.add_argument('--SubSet',help="SubSet Number", default='1')
parser.add_argument('--Fraction',help="Fraction", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
########################################     Main body functions    #########################################
args = parser.parse_args()
SubSet=args.SubSet
fraction=args.Fraction
AFS_DIR=args.AFS
EOS_DIR=args.EOS
input_track_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E1_TRACKS.csv'
input_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E2_E3_RawSeeds_'+SubSet+'_'+fraction+'.csv'
output_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E3_E3_DecoratedSeeds_'+SubSet+'_'+fraction+'.csv'
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
print(UF.TimeStamp(),'Beginning the decorating part...')
for s in range(0,limit):
    seed=Seed(seeds.pop(0))
    seed.DecorateTracks(tracks)
    try:
       seed.DecorateSeedGeoInfo()
       new_seed=[seed.TrackHeader[0],seed.TrackHeader[1],seed.Vx,seed.Vy,seed.Vz,seed.DOCA,seed.V_Tr[0],seed.V_Tr[1],seed.Tr_Tr,seed.angle]
    except:
       new_seed=[seed.TrackHeader[0],seed.TrackHeader[1],'Fail','Fail','Fail','Fail','Fail','Fail','Fail','Fail']
    GoodSeeds.append(new_seed)
print(UF.TimeStamp(),bcolors.OKGREEN+'The evaluation seed decoration has been completed..'+bcolors.ENDC)
del tracks
del seeds
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
UF.LogOperations(output_seed_file_location,'StartLog',GoodSeeds)
exit()
