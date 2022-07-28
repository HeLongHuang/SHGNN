import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import glob
import torch
import random
import joblib
import logging
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import torch_cluster
import torch.nn.functional as F
from typing import Optional
from einops import rearrange, repeat
from torch_geometric.utils import softmax
from torch_geometric.nn import radius_graph
from torch_geometric.nn import SAGEConv,GraphNorm,LayerNorm
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DSL(nn.Module):
    def __init__(self,
                 r = 10,
                 n = 7,
                 feature_dim = 512,
                 hidden_dim = 256,
                 cell_centroid_dim = 2,
                 tissue_centroid_dim = 2,
                 out_dim = 32
                ):
        super(DSL,self).__init__()
        
        self.r = r
        self.n = n
        
        self.x_cell_attribute_layer = nn.Sequential(
            nn.Linear(feature_dim,feature_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feature_dim // 2,out_dim)
        )
        
        self.x_cell_location_layer = nn.Sequential(
            nn.BatchNorm1d(cell_centroid_dim),
            nn.Linear(cell_centroid_dim,feature_dim // 8),
            nn.LeakyReLU(),
            nn.Linear(feature_dim // 8,out_dim)
        )
        
        self.x_tissue_3_attribute_layer = nn.Sequential(
            nn.Linear(feature_dim,feature_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feature_dim // 2,out_dim)
        )
        
        self.x_tissue_3_location_layer = nn.Sequential(
            nn.BatchNorm1d(tissue_centroid_dim),
            nn.Linear(tissue_centroid_dim,feature_dim // 8),
            nn.LeakyReLU(),
            nn.Linear(feature_dim // 8,out_dim)
        )
        
    def graph_edge(self,x: torch.Tensor, r: float,batch: Optional[torch.Tensor] = None, loop: bool = False,max_num_neighbors: int = 3, flow: str = 'source_to_target',num_workers: int = 1) -> torch.Tensor:

        assert flow in ['source_to_target', 'target_to_source']
        
        edge_index = torch_cluster.radius(x, x, r, batch, batch,
                            max_num_neighbors if loop else max_num_neighbors + 1,
                            num_workers)
        if flow == 'source_to_target':
            row, col = edge_index[1], edge_index[0]
        else:
            row, col = edge_index[0], edge_index[1]

        if not loop:
            mask = row != col
            row, col = row[mask], col[mask]

        return torch.stack([row, col], dim=0)
    
    def forward(self,x_cell,centroids_cell,x_tissue_3,centroids_tissue_3):
        
        x_cell_attribute = self.x_cell_attribute_layer(x_cell) 
        x_cell_location = self.x_cell_location_layer(centroids_cell)
        
        x_tissue_3_attribute = self.x_tissue_3_attribute_layer(x_tissue_3) 
        x_tissue_3_location = self.x_tissue_3_location_layer(centroids_tissue_3)
        
        x_cell_attribute_loaction = torch.cat((x_cell_attribute,x_cell_location),dim = 1)
        x_tissue_3_attribute_loaction = torch.cat((x_tissue_3_attribute,x_tissue_3_location),dim = 1)
      
        batch = torch.zeros(x_cell_attribute_loaction.shape[0],dtype = torch.long).to(device)
        cell_edge = self.graph_edge(x = x_cell_attribute_loaction,r = self.r,batch = batch,max_num_neighbors = self.n)
        
        batch = torch.zeros(x_tissue_3_attribute_loaction.shape[0],dtype = torch.long).to(device)
        tissue_3_edge = self.graph_edge(x = x_tissue_3_attribute_loaction,r = self.r,batch = batch,max_num_neighbors = self.n)
        
        return cell_edge,tissue_3_edge
    
# cell + tissue3
class GCN(nn.Module):
    def __init__(self,
                Dropout = 0.25,
                Classes = 7,
                FeatureDim = 512,
                ConvHiddenDim = 256,
                ConvOutDim = 256,
                EncoderLayer = 2,
                EncoderHead = 8,
                EncoderDim = 256,
                PoolMethod1 = "mean",
                LocationOutDim = 32
                ):
        super(GCN,self).__init__()   
        assert PoolMethod1 in ["mean","add","max"]
           
        self.conv1 = SAGEConv(in_channels=FeatureDim,out_channels=ConvHiddenDim)          
        self.conv2 = SAGEConv(in_channels=ConvHiddenDim,out_channels=ConvHiddenDim)
        self.conv3 = SAGEConv(in_channels=ConvHiddenDim,out_channels=ConvOutDim)

        self.conv4 = SAGEConv(in_channels=FeatureDim,out_channels=ConvHiddenDim)          
        self.conv5 = SAGEConv(in_channels=ConvHiddenDim,out_channels=ConvHiddenDim)
        self.conv6 = SAGEConv(in_channels=ConvHiddenDim,out_channels=ConvOutDim)
        
        
        self.dsl = DSL(r = 10,n = 7,feature_dim = FeatureDim,cell_centroid_dim = 2,tissue_centroid_dim = 2,out_dim = LocationOutDim)
    
        self.relu = torch.nn.LeakyReLU() 
        self.dropout=nn.Dropout(p=Dropout)         
        
        if PoolMethod1 == "mean":
            self.pool_method_1 = global_mean_pool
        elif PoolMethod1 == "max": 
            self.pool_method_1 = global_max_pool  
        elif PoolMethod1 == "add":
            self.pool_method_1 = global_add_pool

        self.lin1 = torch.nn.Linear(ConvOutDim,ConvOutDim // 2)
        self.lin2 = torch.nn.Linear(ConvOutDim // 2,Classes)
        
        self.norm1 = GraphNorm(ConvHiddenDim)
        self.norm2 = LayerNorm(ConvOutDim // 2)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=ConvOutDim, nhead=EncoderHead,dim_feedforward=EncoderDim,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=EncoderLayer)
        
        self.num_levels = 2
        self.pool = "cls"
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_levels + 1, ConvOutDim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, ConvOutDim))
        self.vit_dropout=nn.Dropout(p=0.2) 
    
        self.attention_layer = nn.Sequential(
            nn.Linear(ConvOutDim,ConvOutDim // 2),
            nn.LeakyReLU(),
            nn.Linear(ConvOutDim // 2 ,1)
        )
        
        
    def forward(self,data):
        
        x_cell,x_tissue_3,assignment_matrix_3,centroids_cell,centroids_tissue_3 = data.x_cell,data.x_tissue_3,data.assignment_matrix_3,data.centroids_cell,data.centroids_tissue_3
        
        # DSL
        edge_index_cell,edge_index_tissue_3 = self.dsl(x_cell,centroids_cell,x_tissue_3,centroids_tissue_3)
        
        # cell path
        x_cell_cov = x_cell

        x_cell_cov = self.conv1(x_cell_cov,edge_index_cell)
        x_cell_cov = self.norm1(x_cell_cov) 
        x_cell_cov = self.relu(x_cell_cov)  
        x_cell_cov = self.dropout(x_cell_cov)

        x_cell_cov = self.conv2(x_cell_cov,edge_index_cell)
        x_cell_cov = self.norm1(x_cell_cov) 
        x_cell_cov = self.relu(x_cell_cov)  
        x_cell_cov = self.dropout(x_cell_cov)


        x_cell_cov = self.conv3(x_cell_cov,edge_index_cell)
        x_cell_cov = self.norm1(x_cell_cov) 
        x_cell_cov = self.relu(x_cell_cov)  
        x_cell_cov = self.dropout(x_cell_cov)


        # tissue_3 path
        x_tissue_3_conv = x_tissue_3

        x_tissue_3_conv = self.conv4(x_tissue_3_conv,edge_index_tissue_3)
        x_tissue_3_conv = self.norm1(x_tissue_3_conv) 
        x_tissue_3_conv = self.relu(x_tissue_3_conv)  
        x_tissue_3_conv = self.dropout(x_tissue_3_conv)

        x_tissue_3_conv = self.conv5(x_tissue_3_conv,edge_index_tissue_3)
        x_tissue_3_conv = self.norm1(x_tissue_3_conv) 
        x_tissue_3_conv = self.relu(x_tissue_3_conv)  
        x_tissue_3_conv = self.dropout(x_tissue_3_conv)

        x_tissue_3_conv = self.conv6(x_tissue_3_conv,edge_index_tissue_3)
        x_tissue_3_conv = self.norm1(x_tissue_3_conv) 
        x_tissue_3_conv = self.relu(x_tissue_3_conv)  
        x_tissue_3_conv = self.dropout(x_tissue_3_conv)
  
        batch = torch.where(assignment_matrix_3 == 1)[1].to(device)

        x_tissue_3_for_cell = x_tissue_3_conv[batch].unsqueeze(1)
        x_cell_cov = x_cell_cov.unsqueeze(1)
        
        
        #Vision transformer
        x_cell_tissue = torch.cat((x_cell_cov,x_tissue_3_for_cell),dim = 1)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x_cell_tissue.shape[0])
        x_cell_tissue = torch.cat((cls_tokens, x_cell_tissue), dim=1)
        x_cell_tissue += self.pos_embedding[:, :(self.num_levels + 1)]
        x_cell_tissue = self.vit_dropout(x_cell_tissue)
        x_cell_tissue = self.transformer_encoder(x_cell_tissue)
        x_cell_tissue = x_cell_tissue.mean(dim = 1) if self.pool == "mean" else x_cell_tissue[:,0]

        batch = batch.new_zeros(x_cell_tissue.size(0))
        attention = self.attention_layer(x_cell_tissue)
        attention = softmax(attention, batch, num_nodes=batch.shape[0])

        batch = batch.new_zeros(x_cell_tissue.size(0))

        x = self.pool_method_1(x_cell_tissue,batch)
        x = self.lin1(x)
        x = self.relu(x)  
        x = self.norm2(x)     
        x = self.dropout(x) 
        x = self.lin2(x)
        
        return x,edge_index_cell,attention


def return_fold_data_path_list():
    train_val_test_dict = joblib.load("../DataSplit/BRACS_3foldcv.pkl")
     
    fold1_train = train_val_test_dict["fold1_train"]
    fold1_val = train_val_test_dict["fold1_val"]
    fold1_test = train_val_test_dict["fold1_test"]
      
    fold2_train = train_val_test_dict["fold2_train"]
    fold2_val = train_val_test_dict["fold2_val"]
    fold2_test = train_val_test_dict["fold2_test"]
    
    fold3_train = train_val_test_dict["fold3_train"]
    fold3_val = train_val_test_dict["fold3_val"]
    fold3_test = train_val_test_dict["fold3_test"]    
            
    return fold1_train,fold1_val,fold1_test,fold2_train,fold2_val,fold2_test,fold3_train,fold3_val,fold3_test


def return_data_list_with_batch(data_path_list,batch_size):
    data_array = []
    for item in data_path_list:
        data_array.append(item)
    random.shuffle(data_array)
    data_array_temp = [data_array[i:i+batch_size] for i in range(0,len(data_array),batch_size)]
    return data_array_temp
 

def train_val_block(data_with_batch,model,optimizer,loss_fun,device,is_train_val_test,fold_num,epoch_num,scheduler):
    
    assert (is_train_val_test == "train" or is_train_val_test == "val")
    
    time_start = time.time()
    
    if is_train_val_test == "train":
        train_total_loss = 0
        train_possibility_array = None
        train_prediction_array = None
        train_target_array = None
        model.train()
        for index,batch_data in enumerate(data_with_batch):   
            train_batch_loss = 0
            for pyg_data in batch_data:
                pyg_data = joblib.load("../PYG_Data/BRACS/" + pyg_data)
                target = torch.tensor([pyg_data.label]).to(device)
                temp_pyg_data = pyg_data.to(device)
                output,edge_index_cell,attention = model(temp_pyg_data)
                train_step_loss = loss_fun(output,target)
                train_batch_loss = train_batch_loss + train_step_loss
                train_total_loss = train_total_loss + train_step_loss
                _, prediction = torch.max(output, 1)
                if train_possibility_array == None:
                    train_possibility_array = output.detach().cpu()
                else:
                    train_possibility_array = torch.cat((train_possibility_array,output.detach().cpu()),dim = 0)
                if train_prediction_array == None:
                    train_prediction_array = prediction.detach().cpu()
                else:
                    train_prediction_array = torch.cat((train_prediction_array,prediction.detach().cpu()),dim = 0)
                if train_target_array == None:
                    train_target_array = target.cpu()
                else:
                    train_target_array = torch.cat((train_target_array,target.detach().cpu()),dim = 0)
                temp_pyg_data = temp_pyg_data.cpu()
                target = target.cpu()
#                 torch.cuda.empty_cache()
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
        train_acc = accuracy_score(train_prediction_array,train_target_array)
        enc = OneHotEncoder()
        target_onehot = enc.fit_transform(train_target_array.unsqueeze(1))
        target_onehot = target_onehot.toarray()
        train_macro_auc = roc_auc_score(np.round(np.array(target_onehot), 0),train_possibility_array , average = "macro", multi_class = "ovo")
        class_report = metrics.classification_report(train_target_array,train_prediction_array, output_dict=True)
        info_dict = {}
        info_dict["train_acc"] = train_acc
        info_dict["train_macro_auc"] = train_macro_auc
        info_dict["train_total_loss"] = train_total_loss.cpu()
        info_dict["class_report"] = class_report
        info_dict["train_possibility_array"] = train_possibility_array
        info_dict["train_prediction_array"] = train_prediction_array
        info_dict["train_target_array"] = train_target_array
        info_dict["train_one_epoch_time"] = time.time() - time_start
        return info_dict  
    
    
    if is_train_val_test == "val":
        val_total_loss = 0
        val_possibility_array = None
        val_prediction_array = None
        val_target_array = None
        model.eval()
        with torch.no_grad():
            for index,batch_data in enumerate(data_with_batch):
                for pyg_data in batch_data:
                    pyg_data = joblib.load("../PYG_Data/BRACS/" + pyg_data)
                    temp_pyg_data = pyg_data.to(device)
                    target = torch.tensor([pyg_data.label]).to(device)
                    output,edge_index_cell,attention = model(temp_pyg_data)
                    val_step_loss = loss_fun(output,target)
                    val_total_loss = val_total_loss + val_step_loss
                    _, prediction = torch.max(output, 1)
                    if val_possibility_array == None:
                        val_possibility_array = output.detach().cpu()
                    else:
                        val_possibility_array = torch.cat((val_possibility_array,output.detach().cpu()),dim = 0)
                    if val_prediction_array == None:
                        val_prediction_array = prediction.detach().cpu()
                    else:
                        val_prediction_array = torch.cat((val_prediction_array,prediction.detach().cpu()),dim = 0)
                    if val_target_array == None:
                        val_target_array = target.cpu()
                    else:
                        val_target_array = torch.cat((val_target_array,target.cpu()),dim = 0)  
                    temp_pyg_data = temp_pyg_data.cpu()
                    target = target.cpu()
#                     torch.cuda.empty_cache()
            val_acc = accuracy_score(val_prediction_array,val_target_array)
            enc = OneHotEncoder()
            target_onehot = enc.fit_transform(val_target_array.unsqueeze(1))
            target_onehot = target_onehot.toarray()
            val_macro_auc = roc_auc_score(np.round(np.array(target_onehot), 0),val_possibility_array , average = "macro", multi_class = "ovo")
            class_report = metrics.classification_report(val_target_array,val_prediction_array, output_dict=True)
            info_dict = {}
            info_dict["val_acc"] = val_acc
            info_dict["val_macro_auc"] = val_macro_auc
            info_dict["val_total_loss"] = val_total_loss.cpu()
            info_dict["class_report"] = class_report
            info_dict["val_possibility_array"] = val_possibility_array
            info_dict["val_prediction_array"] = val_prediction_array
            info_dict["val_target_array"] = val_target_array
            info_dict["val_one_epoch_time"] = time.time() - time_start
            return info_dict
    
def print_info(fold_num,folds,epoch_num,epochs,info_dict,model,logger):
    if model == "train": 
        logger.info("\n Train:Fold [{}/{}]\n Epoch [{}/{}]\n TrainTotalLoss: {}\n Train_Macro_Auc: {}\n Train_Weighted_F1: {}\n Train_One_Epoch_Time: {}\n".format(
            fold_num + 1,
            folds,
            epoch_num + 1,
            epochs,
            info_dict["train_total_loss"].item(),
            info_dict["train_macro_auc"],
            info_dict["class_report"]["weighted avg"]["f1-score"],
            info_dict["train_one_epoch_time"]
        ))
    if model == "val":
        logger.info("\n Val:Fold [{}/{}]\n Epoch [{}/{}]\n ValTotalLoss: {}\n Val_Macro_Auc: {}\n Val_Weighted_F1: {}\n Val_One_Epoch_Times: {}\n\n".format(
            fold_num + 1,
            folds,
            epoch_num + 1,
            epochs,
            info_dict["val_total_loss"].item(),
            info_dict["val_macro_auc"],
            info_dict["class_report"]["weighted avg"]["f1-score"],
            info_dict["val_one_epoch_time"]
        ))

def main(args):
    
    Epoch = args.Epoch
    BatchSize = args.BatchSize
    LR = args.LR
    Dropout = args.Dropout
    Classes = args.Classes
    FeatureDim = args.FeatureDim
    ConvHiddenDim = args.ConvHiddenDim
    ConvOutDim = args.ConvOutDim
    EncoderLayer = args.EncoderLayer
    EncoderHead = args.EncoderHead
    EncoderDim = args.EncoderDim
    PoolMethod1 = args.PoolMethod1
    LocationOutDim = args.LocationOutDim
 
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("../Logs/BRACS_3foldcv.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    fold1_train_data_path_list,fold1_val_data_path_list,fold1_test_data_path_list,fold2_train_data_path_list,fold2_val_data_path_list,fold2_test_data_path_list,fold3_train_data_path_list,fold3_val_data_path_list,fold3_test_data_path_list = return_fold_data_path_list()
        
    fold_num = 3   
    for fold_num_temp in range(fold_num):
        if fold_num_temp == 0:
            fold_train_data_with_batch = return_data_list_with_batch(fold1_train_data_path_list,BatchSize)
            fold_val_data_with_batch = return_data_list_with_batch(fold1_val_data_path_list,BatchSize)
            fold_test_data_with_batch = return_data_list_with_batch(fold1_test_data_path_list,BatchSize)
        if fold_num_temp == 1:
            fold_train_data_with_batch = return_data_list_with_batch(fold2_train_data_path_list,BatchSize)
            fold_val_data_with_batch = return_data_list_with_batch(fold2_val_data_path_list,BatchSize)
            fold_test_data_with_batch = return_data_list_with_batch(fold2_test_data_path_list,BatchSize)
        if fold_num_temp == 2:
            fold_train_data_with_batch = return_data_list_with_batch(fold3_train_data_path_list,BatchSize)
            fold_val_data_with_batch = return_data_list_with_batch(fold3_val_data_path_list,BatchSize)
            fold_test_data_with_batch = return_data_list_with_batch(fold3_test_data_path_list,BatchSize)   
        model = GCN(Dropout = Dropout,
                    Classes = Classes,   
                    FeatureDim = FeatureDim,
                    ConvHiddenDim = ConvHiddenDim,
                    ConvOutDim = ConvOutDim,
                    EncoderLayer = EncoderLayer,
                    EncoderHead = EncoderHead,
                    EncoderDim = EncoderDim,
                    PoolMethod1 = PoolMethod1,
                    LocationOutDim = LocationOutDim
                   ).to(device)
        optimizer=Adam(model.parameters(),lr=LR,weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer,step_size = 40,gamma = 0.1)
        loss_fun = nn.CrossEntropyLoss()
        best_auc = 0
        pbar = tqdm(range(Epoch))
        for epoch_num in pbar:
            time_begin = time.time()
            train_info_dict = train_val_block(fold_train_data_with_batch,model,optimizer,loss_fun,device,"train",fold_num_temp,epoch_num,scheduler)
            print_info(fold_num_temp,fold_num,epoch_num,Epoch,train_info_dict,"train",logger)
            val_info_dict = train_val_block(fold_val_data_with_batch,model,optimizer,loss_fun,device,"val",fold_num_temp,epoch_num,scheduler)
            print_info(fold_num_temp,fold_num,epoch_num,Epoch,val_info_dict,"val",logger)
            scheduler.step()
            time_end = time.time()
            print("TIME CONSUMING --------------------------------> EPOCH:",epoch_num,"TIME:",time_end - time_begin,"S")
            #----------auc----------  
            if val_info_dict["val_macro_auc"] > best_auc:
                best_auc = val_info_dict["val_macro_auc"]
                torch.save(model.state_dict(), "../SavedModels/BRACS_3foldcv/fold{}_best_val_auc_model.pt".format(fold_num_temp + 1))
                
def get_params():
    parser = argparse.ArgumentParser(description='MICCAI')
    parser.add_argument("--Epoch", type=int, default=60, help="")
    parser.add_argument("--BatchSize", type=int, default=30, help="")
    parser.add_argument("--LR", type=float, default=0.0001, help="")
    parser.add_argument("--Dropout", type=float, default=0.25, help="")
    parser.add_argument("--Classes", type=int, default=7, help="")
    parser.add_argument("--FeatureDim", type=int, default=512, help="")
    parser.add_argument("--ConvHiddenDim", type=int, default=256, help="")
    parser.add_argument("--ConvOutDim", type=int, default=256, help="")
    parser.add_argument("--EncoderLayer", type=int, default=2, help="")
    parser.add_argument("--EncoderHead", type=int, default=8, help="")
    parser.add_argument("--EncoderDim", type=int, default=256, help="") 
    parser.add_argument("--PoolMethod1", type=str, default="mean", help="")
    parser.add_argument("--LocationOutDim", type=int, default=32, help="")
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    
    args = get_params()
    main(args)
