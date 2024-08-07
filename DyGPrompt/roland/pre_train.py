import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from roland import DynamicGCNModel
import argparse
import sys
from dataloader import SnapshotNodeDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

DATA = args.data
lr = args.lr
# 
def compareloss(origi,positive,negative,temperature):
    #input:(graphnum,max_n_num,3 target,positive sample , negative sample)
    temperature=torch.tensor(temperature,dtype=float)
 
    result=-1*torch.log(torch.exp(F.cosine_similarity(origi,positive)/temperature)/torch.exp(F.cosine_similarity(origi,negative)/temperature))
    return result.mean()

device = "cuda:0"
DATA = args.data
g_df = pd.read_csv('../processed/ml_{}.csv'.format(DATA))
e_feat = np.load('../processed/ml_{}.npy'.format(DATA))

n_feat = np.load('../processed/ml_{}_node.npy'.format(DATA))

n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
edge_features = torch.nn.Embedding.from_pretrained(e_feat_th, padding_idx=0, freeze=True)
node_features = torch.nn.Embedding.from_pretrained(n_feat_th, padding_idx=0, freeze=True)


src_l = g_df.u.values # 
dst_l = g_df.i.values# 
e_idx_l = g_df.idx.values# 
label_l = g_df.label.values# 
ts_l = g_df.ts.values# 
in_channels = e_feat.shape[1]
feature_dim = node_features.embedding_dim

# 
# 
additional_vectors_count = len(set(dst_l))

additional_vectors = torch.randn(additional_vectors_count, feature_dim)

extended_node_features_weights = torch.cat([node_features.weight, additional_vectors], dim=0)

node_features = torch.nn.Embedding.from_pretrained(extended_node_features_weights, freeze=True)
node_features = node_features.to(device)


start_ts, end_ts = list(np.quantile(g_df.ts, [0.20,0.80]))
#%

df_sorted = g_df[g_df['ts'] <= end_ts]

unique_ts = df_sorted['ts'].unique()

filtered_ts = unique_ts[(unique_ts > start_ts) & (unique_ts < end_ts)]
# 

num_snapshots = min(len(filtered_ts), 40)
selected_indices = np.linspace(0, len(filtered_ts)-1, num=num_snapshots, dtype=int, endpoint=True)
    # 
selected_ts = filtered_ts[selected_indices]
# 
cumulative_snapshots = []

#
for ts in selected_ts:

    snapshot_edges = g_df[g_df['ts'] <= ts]

    # edge_index = torch.tensor([snapshot_edges['u'].values, snapshot_edges['i'].values], dtype=torch.long)
    
    cumulative_snapshots.append((ts, snapshot_edges['u'].values,snapshot_edges['i'].values))


# print(cumulative_snapshots[0][0])

model = DynamicGCNModel(in_channels = in_channels, hidden_channels=in_channels, out_channels=in_channels)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = compareloss
x_prev1 = torch.zeros_like(node_features.weight)
x_prev2 = torch.zeros_like(node_features.weight)

x_prev1_negative = torch.zeros_like(node_features.weight)
x_prev2_negative = torch.zeros_like(node_features.weight)

x_prev1_positive = torch.zeros_like(node_features.weight)
x_prev2_positive = torch.zeros_like(node_features.weight)
temperature = torch.tensor(0.1)
def train(snapshot, model, node_features, optimizer,x_prev1 = x_prev1,x_prev2=x_prev2,x_prev1_negative=x_prev1_negative,x_prev1_positive=x_prev1_positive,x_prev2_positive=x_prev2_positive,x_prev2_negative=x_prev2_negative, batch_size=4096,criterion = compareloss,device=device,temperature = temperature):
    t_loss = []
    ts = snapshot[0]
    #  ts
    ts = torch.tensor(ts,dtype=torch.float32) 
    ts_expanded = ts.expand(node_features.weight.shape[0], 1).to(device)
    
    source_indices = snapshot[1]#
    dest_indices = snapshot[2]#
    # edges = snapshot['edges']
    dataset = SnapshotNodeDataset(node_features, source_indices,dest_indices)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for source_features,dest_features, batch_source,batch_dest in data_loader:       
        optimizer.zero_grad()
       
        # 
        source_indices = batch_source.to(dtype=torch.long)
        dest_indices = batch_dest.to(dtype=torch.long)

        random_negative = np.random.choice(df_sorted['i'].values, size=len(source_indices), replace=False)

        # random_negative = torch.tensor(random_negative, dtype=torch.long, device=device)
        random_negative = torch.tensor(random_negative, dtype=torch.long)
        
        negative_indices = random_negative.to(dtype=torch.long)

        edge_index = torch.stack([source_indices, dest_indices], dim=0).to(device)
        edge_index_positive = torch.stack([dest_indices, source_indices], dim=0).to(device)
        edge_index_negative = torch.stack([source_indices, negative_indices], dim=0).to(device)

  
        optimizer.zero_grad()

    # ori
        H1,H2 = model(node_features.weight,edge_index,x_prev1,x_prev2,ts_expanded)   
        x_prev1 = H1.clone().detach()
        x_prev2 = H2.clone().detach()
        # H1 = model.layer1(node_features.weight, edge_index, x_prev1)
        # H1 = F.relu(H1)
        # H2 = model.layer2(H1, edge_index, x_prev2)
        # H2 += model.skip_conn(node_features.weight)
        # H2 = torch.nn.BatchNorm1d(H2.size(1)).to(H2.device)(H2)

        H1_pos,H2_pos = model(node_features.weight, edge_index_positive, x_prev1_positive, x_prev2_positive,ts_expanded)
        x_prev1_positive = H1_pos.clone().detach()
        x_prev2_positive = H2_pos.clone().detach()  
        # H1_pos = model.layer1(node_features.weight, edge_index_positive, x_prev1_positive)
        # H1_pos = F.relu(H1_pos)
        # H2_pos = model.layer2(H1_pos, edge_index_positive, x_prev2_positive)
        # H2_pos += model.skip_conn(node_features.weight)
        # H2_pos = torch.nn.BatchNorm1d(H2_pos.size(1)).to(H2_pos.device)(H2_pos)
        

        H1_neg,H2_neg = model(node_features.weight,edge_index_negative,x_prev1_negative,x_prev2_negative,ts_expanded)
        x_prev1_negative = H1_neg.clone().detach()
        x_prev2_negative = H2_neg.clone().detach()  

        
        # H1_neg = model.layer1(node_features.weight, edge_index_negative, x_prev1_negative)
        # H1_neg = F.relu(H1_neg)
        # H2_neg = model.layer2(H1_neg, edge_index_negative, x_prev2_negative)
        # H2_neg += model.skip_conn(node_features.weight)
        # H2_neg = torch.nn.BatchNorm1d(H2_neg.size(1)).to(H2_neg.device)(H2_neg)
        
        
        loss = criterion(H2,H2_pos,H2_neg,temperature)
        loss.backward()
        optimizer.step()


        t_loss.append(loss.item())
        # x_prev2[edge_index[0]] = H2.detach()
    x = np.mean(t_loss)
    return x

    

# snapshot_0 = cumulative_snapshots[2]
# train_loss = train(snapshot_0, model,node_features, optimizer,batch_size=8192)
model.train()
for epoch in tqdm(range(50)):
    loss = []
    x_prev1 = torch.zeros_like(node_features.weight).to(device)
    x_prev2 = torch.zeros_like(node_features.weight).to(device)

    x_prev1_negative = torch.zeros_like(node_features.weight).to(device)
    x_prev2_negative = torch.zeros_like(node_features.weight).to(device)

    x_prev1_positive = torch.zeros_like(node_features.weight).to(device)
    x_prev2_positive = torch.zeros_like(node_features.weight).to(device)
    for snapshot in cumulative_snapshots:
            
        train_loss = train(snapshot, model,node_features, optimizer,batch_size=8192)
        loss.append(train_loss)
    print(np.mean(train_loss))

torch.save(model.state_dict(), 'roland_GCN.pth')
