########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import csv
import argparse
import math
import ast
import numpy as np
import logging
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import copy
import pickle
import torch





########################## Visual Formatting #################################################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

########################## Setting the parser ################################################
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Mode',help="Please enter the mode: Create/Test/Train", default='Create')
parser.add_argument('--ImageSet',help="Please enter the image set", default='1')
parser.add_argument('--DNA',help="Please enter the model dna", default='[[4, 4, 1, 2, 2, 2, 2], [5, 4, 1, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [], [3, 4, 2], [3, 4, 2], [2, 4, 2], [], [], [7, 1, 1, 4]]')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--LR',help="Please enter the value of learning rate", default='0.001')
parser.add_argument('--Epoch',help="Please enter the epoch number", default='1')
parser.add_argument('--EpochLength',help="Please enter the epoch number", default='1000')
parser.add_argument('--ModelName',help="Name of the model", default='2T_100_MC_1_model')
parser.add_argument('--ModelNewName',help="Name of the model", default='2T_100_MC_1_model')
parser.add_argument('--f',help="Image set location (for test)", default='')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ImageSet=args.ImageSet
Mode=args.Mode
DNA=ast.literal_eval(args.DNA)
HiddenLayerDNA=[]
FullyConnectedDNA=[]
OutputDNA=[]
if args.LR == 'Default':
    LR = 0.001
else:
    LR = float(args.LR)
# for gene in DNA:
#     if DNA.index(gene)<=4 and len(gene)>0:
#         HiddenLayerDNA.append(gene)
#     elif DNA.index(gene)<=9 and len(gene)>0:
#         FullyConnectedDNA.append(gene)
#     elif DNA.index(gene)>9 and len(gene)>0:
#         OutputDNA.append(gene)
#
# act_fun_list=['N/A','linear','exponential','elu','relu', 'selu','sigmoid','softmax','softplus','softsign','tanh']
# ValidModel=True
# Accuracy=0.0
# Accuracy0=0.0
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'EDER-VIANN'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
flocation=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM4_GM5_TRAIN_SET_'+ImageSet+'.pkl'
if Mode=='Test' and args.f!='':
   vlocation=args.f
else:
   vlocation=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/GM4_GM5_VALIDATION_SET.pkl'

##############################################################################################################################
######################################### Starting the program ################################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising     EDER-VIANN   model creation module #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)




from torch.nn import Linear
from torch.nn import Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TAGConv
from torch_geometric.nn import GMMConv
from torch.optim.lr_scheduler import StepLR

print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
train_file=open(flocation,'rb')
TrainImages=pickle.load(train_file)
train_file.close()
train_dataset = []
for image in TrainImages[:1000]:
    image.GraphSeed.y = image.GraphSeed.y
    train_dataset.append(image.GraphSeed)


print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+vlocation+bcolors.ENDC)
test_file=open(vlocation,'rb')
TestImages=pickle.load(test_file)
test_file.close()
test_dataset = []
for image in TestImages[:100] :
    image.GraphSeed.y = image.GraphSeed.y
    test_dataset.append(image.GraphSeed)

num_node_features = 4
num_classes = 2

class model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(model, self).__init__()
        torch.manual_seed(12345)
        #GCN layers
        self.conv1 = GCNConv(num_node_features , hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        #TAGCN layers
        self.tagconv1 = TAGConv(num_node_features, hidden_channels)
        self.tagconv2 = TAGConv(hidden_channels, hidden_channels)
        self.tagconv3 = TAGConv(hidden_channels, hidden_channels)
        
        #GMMConv layers
        #self.gmmconv1 = GMMConv(num_node_features, hidden_channels, 3, 1)
        #self.gmmconv2 = GMMConv(hidden_channels, hidden_channels, 3, 1)
        #self.gmmconv3 = GMMConv(hidden_channels, hidden_channels, 3, 1)
        
        self.lin = Linear(hidden_channels, num_classes)
        self.softmax = Softmax()

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        #x = self.conv1(x, edge_index)
        x = self.tagconv1(x, edge_index)
        #x = self.mmconv1(x, edge_index)
        x = x.relu()
        
        #x = self.conv2(x, edge_index)
        x = self.tagconv2(x, edge_index)
        #x = self.mmconv2(x, edge_index)
        x = x.relu()
        
        #x = self.conv3(x, edge_index)
        #x = self.tagconv3(x, edge_index)
        #x = self.mmconv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = self.softmax(x)
        return x



#Estimate number of images in the training file
#Calculate number of batches used for this job
#TrainBatchSize=(OutputDNA[0][1]*4)




from torch_geometric.loader import DataLoader
    
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)




def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()
     correct = 0
     loss_accumulative = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         y_index = data.y.argmax(dim=1)
         correct += int((pred == y_index).sum())  # Check against ground-truth labels.
         loss = criterion(out, data.y)
         loss_accumulative += float(loss)
     return (correct / len(loader.dataset), loss_accumulative/len(loader.dataset))  # Derive ratio of correct predictions.
     
#model = GCN(hidden_channels=32)

if Mode=='Create':
 model_name=EOSsubModelDIR+'/'+args.ModelNewName
 model = model(hidden_channels=32)
 optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 criterion = torch.nn.CrossEntropyLoss()
 State_Save_Path=EOSsubModelDIR+'/'+args.ModelNewName+'_State_Save'
 log_name=EOSsubModelDIR+'/'+ args.ModelNewName + '.csv'
 log=[['Epoch', 'Training accuracy', 'testing accuracy', 'Train loss', 'Test loss']]
 for epoch in range(int(args.Epoch), int(args.Epoch)+int(args.EpochLength)):
        train()
        train_acc = test(train_loader)[0]
        train_loss = test(train_loader)[1]
        test_acc = test(test_loader)[0]
        test_loss = test(test_loader)[1]
        log.append([epoch, train_acc, test_acc, train_loss, test_loss])
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f} Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f}')
 torch.save(model.state_dict(), model_name)
 torch.save({'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, State_Save_Path)
 UF.LogOperations(log_name,'StartLog',log)
if Mode=='Train':
 model_name=EOSsubModelDIR+'/'+args.ModelName
 model = model(hidden_channels=32)
 model.load_state_dict(torch.load(model_name))
 optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 criterion = torch.nn.CrossEntropyLoss()
 State_Save_Path=EOSsubModelDIR+'/'+args.ModelNewName+'_State_Save'
 checkpoint = torch.load(State_Save_Path)
 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 log_name=EOSsubModelDIR+'/'+ args.ModelName + '.csv'
 log=[]
 for epoch in range(int(args.Epoch), int(args.Epoch)+int(args.EpochLength)):
        train()
        train_acc = test(train_loader)[0]
        train_loss = test(train_loader)[1]
        test_acc = test(test_loader)[0]
        test_loss = test(test_loader)[1]
        log.append([epoch, train_acc, test_acc, train_loss, test_loss])
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f} Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f}')
 torch.save(model.state_dict(), model_name)
 UF.LogOperations(log_name,'UpdateLog',log)


#print(UF.TimeStamp(),'This iteration will be split in',bcolors.BOLD+str(NTrainBatches)+bcolors.ENDC,str(TrainBatchSize),'-size batches')
#exit()

#print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+vlocation+bcolors.ENDC)
#val_file=open(vlocation,'rb')
#ValImages=pickle.load(val_file)
#val_file.close()
#
#print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been loaded successfully..."+bcolors.ENDC)
#
#NValBatches=math.ceil(float(len(ValImages))/float(TrainBatchSize))
#
#print(UF.TimeStamp(),'Loading the model...')
###### This but has to be converted to a part that interprets DNA code  ###################################
#if args.LR=='Default':
#  LR=10**(-int(OutputDNA[0][3]))
#  opt = adam(learning_rate=10**(-int(OutputDNA[0][3])))
#else:
#    LR=float(args.LR)
#    opt = adam(learning_rate=float(args.LR))
# if Mode=='Train':
#            model_name=EOSsubModelDIR+'/'+args.ModelName
#            model=tf.keras.models.load_model(model_name)
#            K.set_value(model.optimizer.learning_rate, LR)
#            model.summary()
#            print(model.optimizer.get_config())
# if Mode!='Train' and Mode!='Test':
#            #try:
#              model = Sequential()
#              for HL in HiddenLayerDNA:
#                  Nodes=HL[0]*16
#                  KS=(HL[2]*2)+1
#                  PS=HL[3]
#                  DR=float(HL[6]-1)/10.0
#                  if HiddenLayerDNA.index(HL)==0:
#                     model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform', input_shape=(TrainImages[0].H,TrainImages[0].W,TrainImages[0].L,1)))
#                  else:
#                     model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform'))
#                  if PS>1:
#                     model.add(MaxPooling3D(pool_size=(PS, PS, PS)))
#                  model.add(BatchNormalization(center=HL[4]>1, scale=HL[5]>1))
#                  model.add(Dropout(DR))
#              model.add(Flatten())
#              for FC in FullyConnectedDNA:
#                      Nodes=4**FC[0]
#                      DR=float(FC[2]-1)/10.0
#                      model.add(Dense(Nodes, activation=act_fun_list[FC[1]], kernel_initializer='he_uniform'))
#                      model.add(Dropout(DR))
#              model.add(Dense(2, activation=act_fun_list[OutputDNA[0][0]]))
 # Compile the model
            #  model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            #  model.summary()
            #  print(model.optimizer.get_config())
           #  exit()
           #except:
           #   print(UF.TimeStamp(), bcolors.FAIL+"Invalid model, aborting the training..."+bcolors.ENDC)
           #   ValidModel=False
            #  exit()
# if Mode=='Test':
#            model_name=EOSsubModelDIR+'/'+args.ModelName
#            model=tf.keras.models.load_model(model_name)
#            K.set_value(model.optimizer.learning_rate, LR)
#            model.summary()
#            print(model.optimizer.get_config())
#            for ib in range(0,NValBatches):
#               StartSeed=(ib*TrainBatchSize)+1
#               EndSeed=StartSeed+TrainBatchSize-1
#               BatchImages=UF.LoadRenderImages(ValImages,StartSeed,EndSeed)
#               a=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
#               val_loss=a[0]
#               val_acc=a[1]
#               progress=int(round((float(ib)/float(NValBatches))*100,0))
#               print("Validation in progress ",progress,' %',"Validation loss is:",val_loss,"Validation accuracy is:",val_acc , end="\r", flush=True)
#            print('Test is finished')
#            print("Final Validation loss is:",val_loss)
#            print("Final Validation accuracy is:",val_acc)
#            exit()
#records=[]
#print(UF.TimeStamp(),'Starting the training process... ')
#for ib in range(0,NTrainBatches):
#    StartSeed=(ib*TrainBatchSize)+1
#    EndSeed=StartSeed+TrainBatchSize-1
#    BatchImages=UF.LoadRenderImages(TrainImages,StartSeed,EndSeed)
#    model.train_on_batch(BatchImages[0],BatchImages[1])
#    progress=int(round((float(ib)/float(NTrainBatches))*100,0))
#    print("Training in progress ",progress,' %', end="\r", flush=True)
#print(UF.TimeStamp(),'Finished with the training... ')
#print(UF.TimeStamp(),'Evaluating this epoch ')
#model.reset_metrics()
#for ib in range(0,NTrainBatches):
#    StartSeed=(ib*TrainBatchSize)+1
#    EndSeed=StartSeed+TrainBatchSize-1
#    BatchImages=UF.LoadRenderImages(TrainImages,StartSeed,EndSeed)
#    t=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
#    train_loss=t[0]
#    train_acc=t[1]
#model.reset_metrics()
#for ib in range(0,NValBatches):
#    StartSeed=(ib*TrainBatchSize)+1
#    EndSeed=StartSeed+TrainBatchSize-1
#    BatchImages=UF.LoadRenderImages(ValImages,StartSeed,EndSeed)
#    a=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
#    val_loss=a[0]
#    val_acc=a[1]
#if ValidModel:
#    model_name=EOSsubModelDIR+'/'+args.ModelNewName
#    model.save(model_name)
#    records.append([int(args.Epoch),ImageSet,len(TrainImages),train_loss,train_acc,val_loss,val_acc])
#    UF.LogOperations(EOSsubModelDIR+'/'+'GM5_GM5_model_train_log_'+ImageSet+'.csv','StartLog', records)

