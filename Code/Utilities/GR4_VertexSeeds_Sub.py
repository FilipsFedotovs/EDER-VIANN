#This simple script prepares data for CNN
########################################    Import libraries    #############################################
#import csv
import Utility_Functions as UF
import argparse
import pandas as pd #We use Panda for a routine data processing
import pickle
import torch
from torch.nn import Linear
from torch.nn import Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TAGConv
from torch_geometric.nn import GMMConv
from torch_geometric.loader import DataLoader
# import tensorflow as tf
# from tensorflow import keras

import os, psutil #helps to monitor the memory
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
parser.add_argument('--Set',help="Set Number", default='1')
parser.add_argument('--Fraction',help="Fraction", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--resolution',help="Resolution in microns per pixel", default='100')
parser.add_argument('--acceptance',help="Vertex fit minimum acceptance", default='0.5')
parser.add_argument('--MaxX',help="Image size in microns along the x-axis", default='3500.0')
parser.add_argument('--MaxY',help="Image size in microns along the y-axis", default='1000.0')
parser.add_argument('--MaxZ',help="Image size in microns along the z-axis", default='20000.0')
parser.add_argument('--ModelName',help="Name of the CNN model", default='2T_100_MC_1_model')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
fraction=str(int(args.Fraction))
resolution=float(args.resolution)
acceptance=float(args.acceptance)
#Maximum bounds on the image size in microns
MaxX=float(args.MaxX)
MaxY=float(args.MaxY)
MaxZ=float(args.MaxZ)
#Converting image size bounds in line with resolution settings
AFS_DIR=args.AFS
EOS_DIR=args.EOS
input_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R3_R4_FilteredSeeds_'+Set+'_'+fraction+'.pkl'
output_seed_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R4_R4_CNN_Fit_Seeds_'+Set+'_'+fraction+'.pkl'
print(UF.TimeStamp(),'Analysing the data')
seeds_file=open(input_seed_file_location,'rb')
seeds=pickle.load(seeds_file)
seeds_file.close()
limit=len(seeds)
seed_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
print(UF.TimeStamp(),'Loading the model...')
#Load the model
num_node_features = 4
num_classes = 2
class model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(model, self).__init__()
        torch.manual_seed(12345)
        #TAGCN layers
        self.tagconv1 = TAGConv(num_node_features, hidden_channels,K=2)
        self.tagconv2 = TAGConv(hidden_channels, hidden_channels)
        self.tagconv3 = TAGConv(hidden_channels, hidden_channels)
        # self.conv1 = GCNConv(num_node_features , hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.softmax = Softmax(dim=-1)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        #x = self.conv1(x, edge_index)
        #x = self.tagconv1(x, edge_index)
        x = self.gmmconv1(x, edge_index, edge_attr)
        x = x.relu()

        #x = self.conv2(x, edge_index)
        #x = self.tagconv2(x, edge_index)
        x = self.gmmconv2(x, edge_index, edge_attr)
        x = x.relu()

        #x = self.conv3(x, edge_index)
        #x = self.tagconv3(x, edge_index)
        x = self.gmmconv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = self.softmax(x)
        return x
model_name=EOS_DIR+'EDER-VIANN/Models/'+args.ModelName
print(model_name)
model = model(hidden_channels=16)
model.eval()
print(model)
model.load_state_dict(torch.load(model_name))
print('Here')
#create seeds
GoodSeeds=[]
print(UF.TimeStamp(),'Beginning the vertexing part...')
for s in range(0,limit):
    seed=seeds.pop(0)
    seed.PrepareTrackGraph(MaxX,MaxY,MaxZ,True)
    rec_loader = DataLoader([seed.GraphSeed], batch_size=1, shuffle=False)
    for data in rec_loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    seed.CNNFitSeed(out[0][1].item())
    if seed.Seed_CNN_Fit>=acceptance:
              GoodSeeds.append(seed)
    else:
              continue
print(UF.TimeStamp(),bcolors.OKGREEN+'The vertexing has been completed..'+bcolors.ENDC)
del seeds
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
open_file = open(output_seed_file_location, "wb")
pickle.dump(GoodSeeds, open_file)
open_file.close()
exit()
