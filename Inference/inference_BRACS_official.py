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
from sklearn.metrics import classification_report
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
    
    
def return_data_list_with_batch(data_path_list,batch_size):
    data_array = []
    for item in data_path_list:
        data_array.append(item)
    data_array_temp = [data_array[i:i+batch_size] for i in range(0,len(data_array),batch_size)]
    return data_array_temp



def test_block(data_with_batch,model,device):
    
    #auc calculation
    test_possibility_array = None
    test_prediction_array = None
    test_target_array = None
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(data_with_batch, desc='Testing', unit='batch'):   
            for pyg_data in batch_data:
#                 print("Inference for", pyg_data)
                pyg_data = joblib.load("../PYG_Data/BRACS/" + pyg_data)
                temp_pyg_data = pyg_data.to(device)
                target = torch.tensor([pyg_data.label]).to(device)
                output,edge_index_cell,attention = model(temp_pyg_data)
                _, prediction = torch.max(output, 1)
                if test_possibility_array == None:
                    test_possibility_array = output.detach().cpu()
                else:
                    test_possibility_array = torch.cat((test_possibility_array,output.detach().cpu()),dim = 0)
                if test_prediction_array == None:
                    test_prediction_array = prediction.cpu()
                else:
                    test_prediction_array = torch.cat((test_prediction_array,prediction.cpu()),dim = 0)
                if test_target_array == None:
                    test_target_array = target.cpu()
                else:
                    test_target_array = torch.cat((test_target_array,target.cpu()),dim = 0)
                temp_pyg_data = temp_pyg_data.cpu()
                target = target.cpu()


    #auc calculation
    enc = OneHotEncoder()
    target_onehot = enc.fit_transform(test_target_array.unsqueeze(1))
    target_onehot = target_onehot.toarray()
    macro_auc = roc_auc_score(np.round(np.array(target_onehot), 0),test_possibility_array , average = "macro", multi_class = "ovo")
    class_0_auc = metrics.roc_auc_score(target_onehot[:,0],test_possibility_array[:,0])
    class_1_auc = metrics.roc_auc_score(target_onehot[:,1],test_possibility_array[:,1])
    class_2_auc = metrics.roc_auc_score(target_onehot[:,2],test_possibility_array[:,2])
    class_3_auc = metrics.roc_auc_score(target_onehot[:,3],test_possibility_array[:,3])
    class_4_auc = metrics.roc_auc_score(target_onehot[:,4],test_possibility_array[:,4])
    class_5_auc = metrics.roc_auc_score(target_onehot[:,5],test_possibility_array[:,5])
    class_6_auc = metrics.roc_auc_score(target_onehot[:,6],test_possibility_array[:,6])

    #F1 calculation
    class_report = metrics.classification_report(test_target_array,test_prediction_array, output_dict=True)
    weighted_f1_score = class_report["weighted avg"]["f1-score"]
    class_0_f1 = class_report["0"]["f1-score"]
    class_1_f1 = class_report["1"]["f1-score"]
    class_2_f1 = class_report["2"]["f1-score"]
    class_3_f1 = class_report["3"]["f1-score"]        
    class_4_f1 = class_report["4"]["f1-score"]
    class_5_f1 = class_report["5"]["f1-score"]
    class_6_f1 = class_report["6"]["f1-score"]     
        
    #print result
    print("\n")
    print("                         AUC  SUMMARY")
    print("auc_for_class_0          ===========>          ",class_0_auc)
    print("auc_for_class_1          ===========>          ",class_1_auc)
    print("auc_for_class_2          ===========>          ",class_2_auc)
    print("auc_for_class_3          ===========>          ",class_3_auc)
    print("auc_for_class_4          ===========>          ",class_4_auc)
    print("auc_for_class_5          ===========>          ",class_5_auc)
    print("auc_for_class_6          ===========>          ",class_6_auc)
    print("macro_auc                ===========>          ",macro_auc)
    print("\n")
    print("                         F1  SUMMARY")   
    print("f1_for_class_0           ===========>          ",class_0_f1)
    print("f1_for_class_1           ===========>          ",class_1_f1)
    print("f1_for_class_2           ===========>          ",class_2_f1)
    print("f1_for_class_3           ===========>          ",class_3_f1)
    print("f1_for_class_4           ===========>          ",class_4_f1)
    print("f1_for_class_5           ===========>          ",class_5_f1)
    print("f1_for_class_6           ===========>          ",class_6_f1)
    print("weighted_f1_score        ===========>         ",weighted_f1_score)
    print("\n")
    
    #print classification_report
    print("                    Classification report")
    print(classification_report(test_target_array,test_prediction_array))
        
    return macro_auc, weighted_f1_score




def main(args):
    #load model path
    Model_path = args.Model_path
    
    
    #load test data
    BatchSize = args.BatchSize
    BRACS_official = joblib.load("../DataSplit/BRACS_official.pkl")
    test_data_path_list = BRACS_official["test"]
    test_data_path_list.sort()         
    test_data_with_batch = return_data_list_with_batch(test_data_path_list,BatchSize)

    
    #load model
    model = GCN(Dropout = args.Dropout,
                Classes = args.Classes,   
                FeatureDim = args.FeatureDim,
                ConvHiddenDim = args.ConvHiddenDim,
                ConvOutDim = args.ConvOutDim,
                EncoderLayer = args.EncoderLayer,
                EncoderHead = args.EncoderHead,
                EncoderDim = args.EncoderDim,
                PoolMethod1 = args.PoolMethod1,
                LocationOutDim = args.LocationOutDim
               ).to(device)
    model.load_state_dict(torch.load(Model_path))
    
    
    #Inference and calculate metrics
    macro_auc, weighted_f1_score = test_block(test_data_with_batch,model,device)

        


def get_params():
    
    parser = argparse.ArgumentParser(description='MICCAI')
    parser.add_argument("--Model_path", type=str, default="../SavedModels/BRACS_official/ex1_best_val_f1_model.pt", help="")
    parser.add_argument("--BatchSize", type=int, default=30, help="")
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
    
