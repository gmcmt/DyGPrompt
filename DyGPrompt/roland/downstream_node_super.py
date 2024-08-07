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
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--name', type=str, default='wiki', help='Prefix to name the result txt')
parser.add_argument('--fn', type=str, default='GCN', help='Prefix to name the result txt')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        # self.act = torch.nn.Tanh()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)
class LR(torch.nn.Module):
    #drop default = 0.3,best = 0.1
    def __init__(self, dim, drop=0):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        # self.act = torch.nn.LeakyReLU()
        
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)
model_path = f'./roland_{args.fn}.pth'
DATA = args.data
lr = args.lr
device = "cuda:0"
g_df = pd.read_csv('../downstream_data/{}/ds_{}.csv'.format(DATA,DATA))
e_feat = np.load('../processed/ml_{}.npy'.format(DATA))
n_feat = np.load('../processed/ml_{}_node.npy'.format(DATA))
n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
edge_features = torch.nn.Embedding.from_pretrained(e_feat_th, padding_idx=0, freeze=True)
node_features = torch.nn.Embedding.from_pretrained(n_feat_th, padding_idx=0, freeze=True)

in_channels = e_feat.shape[1]
feature_dim = node_features.embedding_dim
src_l = g_df.u.values # 
dst_l = g_df.i.values# 
e_idx_l = g_df.idx.values# 
label_l = g_df.label.values# 
ts_l = g_df.ts.values# 


additional_vectors_count = len(set(dst_l))

additional_vectors = torch.randn(additional_vectors_count, feature_dim)

extended_node_features_weights = torch.cat([node_features.weight, additional_vectors], dim=0)

node_features = torch.nn.Embedding.from_pretrained(extended_node_features_weights, freeze=True)

node_features = node_features.to(device)
val_time, test_time = list(np.quantile(g_df.ts, [0.10, 0.20]))
max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())
random.seed(2020)
#
total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)


mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))

mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values

none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

# define the new nodes sets for testing inductiveness of the model
train_node_set = set(train_src_l).union(train_dst_l)
assert(len(train_node_set - mask_node_set) == len(train_node_set))
new_node_set = total_node_set - train_node_set

# select validation and test dataset
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time
is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
nn_val_flag = valid_val_flag * is_new_node_edge
nn_test_flag = valid_test_flag * is_new_node_edge
val_ts_l = ts_l[valid_val_flag]
test_ts_l = ts_l[valid_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]


x_prev1 = torch.zeros_like(node_features.weight)
x_prev2 = torch.zeros_like(node_features.weight)

x_prev1_negative = torch.zeros_like(node_features.weight)
x_prev2_negative = torch.zeros_like(node_features.weight)

x_prev1_positive = torch.zeros_like(node_features.weight)
x_prev2_positive = torch.zeros_like(node_features.weight)
model = DynamicGCNModel(in_channels = in_channels, hidden_channels=in_channels, out_channels=in_channels)
model.load_state_dict(torch.load(model_path),strict=False)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

lr_model = LR(n_feat.shape[1])
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
lr_model = lr_model.to(device)

#
label_flag = 0
task_start_time = 0
task_start_indices = 0
for i in range(len(label_l)):
    if label_l[i]:
        label_flag += 1
        task_start_time = ts_l[i]
        task_start_indices = i
        break
    
def train(snapshot, model, node_features, optimizer,x_prev1 = x_prev1,x_prev2=x_prev2,x_prev1_negative=x_prev1_negative,x_prev1_positive=x_prev1_positive,x_prev2_positive=x_prev2_positive,x_prev2_negative=x_prev2_negative, batch_size=4096,criterion = criterion,device=device):

    t_loss = []
    ts = snapshot[0]

    ts = torch.tensor(ts,dtype=torch.float32) 
    ts_expanded = ts.expand(node_features.weight.shape[0], 1).to(device)
    
    source_indices = snapshot[1]
    dest_indices = snapshot[2]
 
    # edges = snapshot['edges']
    dataset = SnapshotNodeDataset(node_features, source_indices,dest_indices)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batch = 0
    model.train()
    lr_model.train()
    for source_features,dest_features, batch_source,batch_dest in data_loader:
        num_batch+=1 
        optimizer.zero_grad()
        lr_optimizer.zero_grad()

        source_indices = batch_source.to(dtype=torch.long)

        dest_indices = batch_dest.to(dtype=torch.long)


        edge_index = torch.stack([source_indices, dest_indices], dim=0).to(device)
        H1,H2 = model(node_features.weight,edge_index,x_prev1,x_prev2,ts_expanded)   
        x_prev1 = H1.clone().detach()
        x_prev2 = H2.clone().detach()
        label_l_cut = label_l[source_indices]
        label = torch.from_numpy(label_l_cut).float().to(device)
        lr_prob = lr_model(H2).sigmoid()[source_indices]
        loss = criterion(lr_prob, label)
        loss.backward()
        optimizer.step()
        lr_optimizer.step()
        t_loss.append(loss.item())

    x = np.mean(t_loss)
    return x
def valid(snapshot, model, node_features, optimizer,x_prev1 = x_prev1,x_prev2=x_prev2,x_prev1_negative=x_prev1_negative,x_prev1_positive=x_prev1_positive,x_prev2_positive=x_prev2_positive,x_prev2_negative=x_prev2_negative, batch_size=4096,criterion = criterion,device=device):
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
    num_batch = 0
    model.train()
    lr_model.train()
    ap = []
    auc = []
    pred_prob = np.zeros(len(source_indices))
    
    label_base = label_l[0:len(source_indices)]
    s_id = 0
    e_id = 0
    for source_features,dest_features, batch_source,batch_dest in data_loader:
        num_batch+=1 

        optimizer.zero_grad()
        lr_optimizer.zero_grad()
        s_id = e_id
        
        
    
        # 
        source_indices = batch_source.to(dtype=torch.long)
        e_id += len(source_indices)


        dest_indices = batch_dest.to(dtype=torch.long)


        edge_index = torch.stack([source_indices, dest_indices], dim=0).to(device)
        H1, H2 = model(node_features.weight, edge_index, x_prev1, x_prev2, ts_expanded)   
        x_prev1 = H1.clone().detach()
        x_prev2 = H2.clone().detach()
        label_l_cut = label_l[s_id:e_id]
        label = torch.from_numpy(label_l_cut).float().to(device)  # Ensure label is on GPU for computation

        lr_prob = lr_model(H2).sigmoid()[source_indices]
        pred_prob_0 = lr_prob.detach().cpu().numpy()  # Detach, move to CPU, and convert to NumPy
        pred_prob[s_id:e_id] = lr_prob.detach().cpu().numpy()  
    
    auc_roc = roc_auc_score(label_base, pred_prob)  # Compute ROC AUC score


    return np.mean(auc_roc)
def choose_data(ts_set):
    res = []
    for ts in ts_set:
        snapshot_edges = g_df[g_df['ts'] <= ts]        
        res.append((ts, snapshot_edges['u'].values,snapshot_edges['i'].values))
    return res  
test_aps = []
test_aucs = []
test_nn_aps = []
test_nn_aucs = []


num_snapshots = min(len(train_ts_l), 200)
selected_train_indices = np.linspace(task_start_indices+1, len(train_ts_l)-1, num=num_snapshots, dtype=int, endpoint=True)
selected_val_indices = np.linspace(0, len(val_ts_l)-1, num=num_snapshots, dtype=int, endpoint=True)
selected_test_indices = np.linspace(0, len(test_ts_l)-1, 100, dtype=int, endpoint=True)
selected_nn_test_indices = np.linspace(0, len(nn_test_ts_l)-1, 1000, dtype=int, endpoint=True)
selected_train_ts = train_ts_l[selected_train_indices]
selected_val_ts = val_ts_l[selected_val_indices]
selected_test_ts = test_ts_l[selected_test_indices]


selected_nn_test_ts = nn_test_ts_l[selected_nn_test_indices]


train_snapshots = choose_data(selected_train_ts)
# print(train_snapshots)
val_snapshots = choose_data(selected_val_ts)
test_snapshots = choose_data(selected_test_ts)
nn_test_snapshots = choose_data(selected_nn_test_ts)
for i in range(len(train_snapshots)):

    train_snapshot = train_snapshots[i]
    # print(train_snapshot)
    val_snapshot = val_snapshots[i]
    _ = train(train_snapshot, model,node_features, optimizer,batch_size=8192)
    _ = valid(val_snapshot, model,node_features, optimizer,batch_size=8192)
tmp_ap = []
tmp_auc = []
tmp_nn_ap = []
tmp_nn_auc = []
for i in range(len(test_snapshots)):
    print(i)
    test_snapshot = test_snapshots[i]
    nn_test_snapshot = nn_test_snapshots[i]
    test_auc = valid(test_snapshot, model,node_features, optimizer,batch_size=8192)
    # test_nn_ap, test_nn_auc = valid(nn_test_snapshot, model,node_features, optimizer,batch_size=8192)
    # tmp_ap.append(test_ap)
    tmp_auc.append(test_auc)
    # tmp_nn_ap.append(test_nn_ap)
    # tmp_nn_auc.append(test_nn_auc)
# test_aps.append(np.mean(tmp_ap))
test_aucs.append(np.mean(tmp_auc))
# test_nn_aps.append(np.mean(tmp_nn_ap))
# test_nn_aucs.append(np.mean(tmp_nn_auc))
FUNCTION = args.fn
NAME = args.name

folder_path = "./result/super/%s"%(FUNCTION)  
# np.savetxt(f"{folder_path}/{NAME}_aps.txt", [sum(test_aps)/100], fmt='%s')
np.savetxt(f"{folder_path}/{NAME}_node_aucs.txt", test_aucs, fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_nn_aps.txt", [sum(test_nn_aps)/100], fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_nn_aucs.txt", [sum(test_nn_aucs)/100], fmt='%s')

    
        
    


    
    
    
     

