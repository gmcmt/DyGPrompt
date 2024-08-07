import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
import numpy as np

class MergeLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MergeLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  
        torch.nn.init.xavier_normal_(self.fc.weight)
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)  
        x = self.fc(x)
        return x
class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)

class DynamicGCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicGCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.gru = nn.GRUCell(out_channels, out_channels)  # 
    def forward(self, x, edge_index, x_prev):
        H_l_current = self.conv(x, edge_index)

        H_l_updated = self.gru(H_l_current, x_prev)
        return H_l_updated

class DynamicGCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DynamicGCNModel, self).__init__()
        self.layer1 = DynamicGCNLayer(in_channels, hidden_channels)
        self.layer2 = DynamicGCNLayer(hidden_channels, out_channels)
        self.skip_conn = torch.nn.Linear(in_channels, out_channels)
        self.time_encoder = TimeEncode(expand_dim=in_channels)
        self.merge_layer = MergeLayer(in_channels*2,in_channels)
    def forward(self, node_features, edge_index, x_prev1, x_prev2,ts):
        t_embed = self.time_encoder(ts).squeeze(1)

        node_features = self.merge_layer(node_features,t_embed)
        H1 = self.layer1(node_features, edge_index, x_prev1)
        H1 = F.relu(H1)
        H2 = self.layer2(H1, edge_index, x_prev2)
        H2 += self.skip_conn(node_features)
        H2 = torch.nn.BatchNorm1d(H2.size(1)).to(H2.device)(H2)
        return H1,H2
    
#GAT

class DynamicGATLayer(torch.nn.Module):  # 
    def __init__(self, in_channels, out_channels, heads=2, concat=True):
        super(DynamicGATLayer, self).__init__()
        # 
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=concat)
        self.gru = nn.GRUCell(out_channels * heads if concat else out_channels, out_channels)

    def forward(self, x, edge_index, x_prev):
        # 
        H_l_current = self.conv(x, edge_index)

        H_l_updated = self.gru(H_l_current, x_prev)
        return H_l_updated

class DynamicGATModel(torch.nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, concat=True):
        super(DynamicGATModel, self).__init__()
        self.layer1 = DynamicGATLayer(in_channels, hidden_channels, heads=heads, concat=concat)
        self.layer2 = DynamicGATLayer(hidden_channels * heads if concat else hidden_channels, out_channels, heads=1, concat=False)  # 通常在最后一层不拼接
        self.skip_conn = torch.nn.Linear(in_channels, out_channels)
        self.time_encoder = TimeEncode(expand_dim=in_channels)
        self.merge_layer = MergeLayer(in_channels * 2, in_channels)  

    def forward(self, node_features, edge_index, x_prev1, x_prev2, ts):
        t_embed = self.time_encoder(ts).squeeze(1)

        node_features = self.merge_layer(node_features, t_embed)
        H1 = self.layer1(node_features, edge_index, x_prev1)
        H1 = F.relu(H1)
        H2 = self.layer2(H1, edge_index, x_prev2)
        H2 += self.skip_conn(node_features)
        H2 = torch.nn.BatchNorm1d(H2.size(1)).to(H2.device)(H2)
        return H1, H2