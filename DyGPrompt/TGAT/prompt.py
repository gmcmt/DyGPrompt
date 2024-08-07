import torch
import torch.nn as nn
import numpy as np

class node_prompt_layer(nn.Module):
    def __init__(self,input_dim):
        super(node_prompt_layer, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, node_embedding):
        node_embedding=node_embedding*self.weight
        return node_embedding

#struature prompt
class structure_prompt_layer(nn.Module):
    def __init__(self,size,input_dim):
        super(structure_prompt_layer, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(size,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self ,id,node_embedding):
     
        node_embedding=node_embedding + self.weight[id]
        return node_embedding
#temporal prompt
class time_prompt_layer(nn.Module):
    def __init__(self,input_dim):
        super(time_prompt_layer, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, node_embedding):
        node_embedding = node_embedding * self.weight
        
        return node_embedding

class Tprog_prompt_layer(nn.Module):
    def __init__(self,size,input_dim):
        super(Tprog_prompt_layer, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(size,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
        self.meta = META_NET(input_dim*2)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self ,id,time_diff_embedding,node_embedding):
        # time_diff_embedding = torch.mean(time_diff_embedding,dim=1)
        temp = torch.cat((time_diff_embedding, self.weight[id]), dim=1)
        p = self.meta(temp,use_mlp=1)
     
        temp_1 = torch.cat((node_embedding, p), dim=1)
        node_embedding = self.meta(temp_1,use_mlp=2)
        
        return node_embedding
     
class META_NET(nn.Module):
    def __init__(self, input_dim):
        super(META_NET, self).__init__()
        
        self.mlp1 = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, input_dim // 16)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(input_dim // 16, input_dim//2)),
        ]))
        
        self.mlp2 = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, input_dim // 16)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(input_dim // 16, input_dim//2)),
        ]))

    def forward(self, x, use_mlp=1):

        if use_mlp == 1:
            return self.mlp1(x)
        elif use_mlp == 2:
            return self.mlp2(x)

# class graph_prompt_layer(nn.Module):
#     def __init__(self,input_dim):
#         super(graph_prompt_layer, self).__init__()
        
#         self.weight = torch.nn.Parameter(torch.Tensor(1,input_dim))
        
      
        
        
    #     self.max_n_num=input_dim
    #     self.reset_parameters()
        
    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.weight)
        

    # def forward(self, graph_embedding):
    #     graph_embedding = graph_embedding*self.weight
        
    #     return graph_embedding

class graph_prompt_layer(nn.Module):
    def __init__(self,input_dim):
        super(graph_prompt_layer, self).__init__()
        
        self.weight = torch.nn.Parameter(torch.Tensor(1,input_dim))   
        self.max_n_num=input_dim
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        

    def forward(self, graph_embedding):
        graph_embedding = graph_embedding*self.weight
        
        return graph_embedding


from collections import OrderedDict
class metanet(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(metanet,self).__init__()
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, input_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(input_dim // 16, output_dim))
        ]))
        
     