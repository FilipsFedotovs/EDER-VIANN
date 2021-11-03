import pandas as pd
import csv
import argparse
import Utility_Functions as UF
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Fraction',help="Fraction", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--MaxSeedsPerJob',help="Max seeds per job", default='')
args = parser.parse_args()
fraction=int(args.Fraction)
AFS_DIR=args.AFS
EOS_DIR=args.EOS
input_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R4_R5_CNN_Fit_Seeds.csv'
output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_R5_Link_CNN_Fit_Seeds_'+str(fraction)+'.csv'
data=pd.read_csv(input_file_location,usecols=['Track_1','Track_2','Seed_CNN_Fit'])
MaxSeedsPerJob=int(args.MaxSeedsPerJob)
SeedStart=fraction*MaxSeedsPerJob
SeedEnd=min(len(data),(fraction+1)*MaxSeedsPerJob)
seeds=data.loc[SeedStart:SeedEnd-1]
seeds_1=seeds.drop(['Track_2','Seed_CNN_Fit'],axis=1)
seeds_1=seeds_1.rename(columns={"Track_1": "Track_ID"})
seeds_2=seeds.drop(['Track_1','Seed_CNN_Fit'],axis=1)
seeds_2=seeds_2.rename(columns={"Track_2": "Track_ID"})
seed_list=result = pd.concat([seeds_1,seeds_2])
seed_list=seed_list.sort_values(['Track_ID'])
seed_list.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
data_l=pd.merge(seed_list,data , how="inner", left_on=["Track_ID"], right_on=["Track_1"] )
data_l=data_l.drop(['Track_ID'],axis=1)
data_r=pd.merge(seed_list, data , how="inner", left_on=["Track_ID"], right_on=["Track_2"] )
data_r=data_r.drop(['Track_ID'],axis=1)
data=pd.concat([data_l,data_r])
data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(data['Track_1'], data['Track_2'])]
data.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
data.drop(["Seed_ID"],axis=1,inplace=True)
del seeds_1
del seeds_2
del seed_list
del data_l
del data_r
data=data.values.tolist()
seeds=seeds.values.tolist()
for rows in seeds:
    for i in range(4):
       rows.append([])
for seed in seeds:
        for dt in data:
           if (seed[0]==dt[0] and seed[1]!=dt[1]):
              seed[3].append(dt[1])
              seed[5].append(dt[2])
           elif (seed[0]==dt[1] and seed[1]!=dt[0]):
              seed[3].append(dt[0])
              seed[5].append(dt[2])
           if ((seed[1]==dt[0]) and seed[0]!=dt[1]):
              seed[4].append(dt[1])
              seed[6].append(dt[2])
           elif (seed[1]==dt[1] and seed[0]!=dt[0]):
              seed[4].append(dt[0])
              seed[6].append(dt[2])
        CommonSets = list(set(seed[3]).intersection(seed[4]))
        LinkStrength=0.0
        for CS in CommonSets:
            Lindex=seed[3].index(CS)
            Rindex=seed[4].index(CS)
            LinkStrength+=seed[5][Lindex]
            del seed[5][Lindex]
            del seed[3][Lindex]
            del seed[6][Rindex]
            del seed[4][Rindex]
        UnlinkStrength=sum(seed[6])+sum(seed[5])
        seed.append(CommonSets)
        CommonSetsNo= len(CommonSets)
        OrthogonalSets=(len(seed[3])+len(seed[4]))
        del seed[3:8]
        seed.append(CommonSetsNo)
        seed.append(OrthogonalSets)
        seed.append(LinkStrength)
        seed.append(UnlinkStrength)
Header=[['Track_1','Track_2','Seed_CNN_Fit', 'Links', 'AntiLinks', 'Link_Strength', 'AntiLink_Strenth']]
UF.LogOperations(output_file_location,'StartLog', Header)
UF.LogOperations(output_file_location,'UpdateLog', seeds)



