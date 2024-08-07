"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import random
import argparse
from itertools import chain
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score,precision_score

from module import TGAN
from graph import NeighborFinder
from prompt import Tprog_prompt_layer

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
"""
class LR(torch.nn.Module):
    #drop default = 0.3,best = 0.1
    def __init__(self, dim, drop=0.2):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 100)
        self.fc_2 = torch.nn.Linear(100, 50)
        self.fc_3 = torch.nn.Linear(50, 10)
        self.fc_4 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        x = self.act(self.fc_3(x))
        x = self.dropout(x)
        return self.fc_4(x).squeeze(dim=1)
"""


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=50, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

parser.add_argument('--train_percent', type=float, default=0.05)
parser.add_argument('--val_percent', type=float, default=0.05)
parser.add_argument('--task_num', type=int, default=100)
parser.add_argument('--train_shot_num', type=int, default=3)
parser.add_argument('--val_shot_num', type=int, default=3)
parser.add_argument('--test_shot_num', type=int, default=100)
parser.add_argument('--name', type=str, default='', help='Prefix to name the result txt')
parser.add_argument('--fn', type=str, default='', help='Prefix to name the result txt')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
TRAIN_DATA = args.train_percent
VAL_DATA = args.val_percent
TASK_NUM = args.task_num

TRAIN_SHOT_NUM = args.train_shot_num
VAL_SHOT_NUM = args.val_shot_num
TEST_SHOT_NUM = args.test_shot_num
### set up logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
def eval_epoch(src_l, dst_l, ts_l, label_l, lr_model, tgan, prompt, shot_num,num_layer=NODE_LAYER):
    
    loss = 0
    bs = None
    with torch.no_grad():
        lr_model.eval()
        prompt.eval()
        tgan.eval()
        
        if shot_num:
            indices_1 = random.sample(range(0, 10),shot_num)
            indices_0 = random.sample(range(10,len(src_l)),shot_num*5)
            indices = indices_1 + indices_0
        else:
            bs = 500
            # indices_1 = random.sample(set(np.where(label_l == 1)[0]), 10)
            # indices_0 = random.sample(set(np.where(label_l == 0)[0]), 10*5)
            # indices = indices_1 + indices_0
        if not bs:
            
            src_l_cut = src_l[indices]
            dst_l_cut = dst_l[indices]
            ts_l_cut = ts_l[indices]
            label_l_cut = label_l[indices]
        

            # print(shot_num)
            # print(src_l_cut)
            # print(ts_l_cut)
            src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            _,  _ ,src_t_ngh = full_ngh_finder.get_temporal_neighbor(src_l_cut, ts_l_cut, num_neighbors=1)
            src_t_ngh = src_t_ngh.reshape(1, len(src_t_ngh)) 
            delta_ts = ts_l_cut - src_t_ngh
            d_t = torch.from_numpy(delta_ts).float().to(device)
            # print(delta_ts)
            delta_ts_embed = tgan.time_encoder(d_t).squeeze(0)
                
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            # embedding = prompt(src_embed)
        
            embedding = prompt(src_l_cut,delta_ts_embed,src_embed)   
            src_label = torch.from_numpy(label_l_cut).float().to(device)
                

            # embedding = prompt(src_embed)
            # embedding = src_embed

            lr_prob = lr_model(embedding).sigmoid()

                
            loss += lr_criterion_eval(lr_prob, src_label).item()
            # pred_prob = np.zeros(len(src_l_cut))
            pred_prob = lr_prob.cpu().numpy()
            pred_label = pred_prob > 0.5
            auc_roc = roc_auc_score(label_l_cut, pred_prob)
            acc = (pred_label == label_l_cut).mean()
            f1 = f1_score(label_l_cut, pred_label, average='binary')
            return auc_roc, acc, loss / len(indices),f1
        else:
            num_instance = len(src_l)
            pred_prob = np.zeros(num_instance)
            num_batch = math.ceil(num_instance / bs)
            for k in range(num_batch):          
                s_idx = k * bs
                e_idx = min(num_instance - 1, s_idx + bs)
                src_l_cut = src_l[s_idx:e_idx]
                dst_l_cut = dst_l[s_idx:e_idx]
                indices = [i for i in range(s_idx, e_idx)]

                ts_l_cut = ts_l[s_idx:e_idx]
                label_l_cut = label_l[s_idx:e_idx]
                size = len(src_l_cut)
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
                _,  _ ,src_t_ngh = full_ngh_finder.get_temporal_neighbor(src_l_cut, ts_l_cut, num_neighbors=1)
                src_t_ngh = src_t_ngh.reshape(1, len(src_t_ngh)) 
                delta_ts = ts_l_cut - src_t_ngh
                d_t = torch.from_numpy(delta_ts).float().to(device)
                # print(delta_ts)
                delta_ts_embed = tgan.time_encoder(d_t).squeeze(0)
            
                
                src_label = torch.from_numpy(label_l_cut).float().to(device)
                # embedding = prompt(src_embed)
                # embedding = src_embed
                embedding = prompt(src_l_cut,delta_ts_embed,src_embed)     
       
      
    
                lr_prob = lr_model(embedding).sigmoid()
                loss += lr_criterion_eval(lr_prob, src_label).item()
                pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
            pred_label = pred_prob > 0.5
            acc = (pred_label == label_l).mean()
            auc_roc = roc_auc_score(label_l, pred_prob)
            f1 = f1_score(label_l, pred_label, average='binary')
            return auc_roc, acc, loss / num_instance,f1

            
### Load data and train val test split
g_df = pd.read_csv('./downstream_data/{}/ds_{}.csv'.format(DATA,DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [TRAIN_DATA, TRAIN_DATA+VAL_DATA]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values



label_flag = 0
task_start_time = 0
task_end_time = 0

for i in range(len(label_l)):
    if label_l[i]:
        label_flag += 1
    if label_flag ==20 and task_start_time == 0:
        task_start_time = ts_l[i]

    
    if label_flag ==24:
         task_end_time = ts_l[i]
         break


        
task_start = ts_l >= task_start_time
task_end =  ts_l <= task_end_time
task_time_p = task_start * task_end
task_time_pool = ts_l[task_time_p]
# count = 0 
# for ele in task_time_p:
#     if ele:
#         count +=1
# print("zzzzzzzzzz")
# print(count)
# print(len(task_time_pool))
test_indices = (1 - task_time_p) > 0

test_indices = (1 - task_start) > 0
label_pool = label_l[test_indices]
count = 0 
for i in range(len(label_l)):
    if (ts_l[i]>task_end_time or ts_l[i]<task_start_time) and label_l[i]:
        count +=1





task_time_set = random.sample(set(task_time_pool),TASK_NUM)
np.savetxt("wiki_task_time", task_time_set, fmt='%s')

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())
total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))


# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

device = torch.device('cuda:{}'.format(GPU))
model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'

# for count in range(TASK_NUM):
total_auc = []
total_acc = []
total_f1 = []
path_model = './saved_models/{}_wiki_node_class.pth'.format(DATA)
path_prompt =  './saved_models/{}_prompt_node.pth'.format(DATA)
path_time_prompt =  './saved_models/{}_time_prompt_node.pth'.format(DATA)

for task in tqdm(range(TASK_NUM)):  

    time_stamp = task_time_set[task]
    ts_flag = (ts_l <= time_stamp)
    
    index = np.where(ts_l == time_stamp)[0][0]
    
    
    ts_label_flag_1 = (ts_flag) * (label_l)

    ts_label_flag_1 = ts_label_flag_1[0:index+1]
    record = {}

    for i in range(len(ts_label_flag_1)-1,-1,-1):
        if src_l[i] in record:
    
            # choose_node_flag[i] = 0
            ts_label_flag_1[i] = -1
        else:
            record[src_l[i]] = 1 
            
    num_indices = 10  

    train_indices_1 = random.sample(set(np.where(ts_label_flag_1 == 1)[0]), num_indices)
    train_indices_0 = random.sample(set(np.where((ts_label_flag_1 == 0))[0]), num_indices*5)
    
    ts_label_flag_1[train_indices_1], ts_label_flag_1[train_indices_0] = -1, -2
    
    val_indices_1 = random.sample(set(np.where(ts_label_flag_1 == 1)[0]), num_indices)
    val_indices_0 = random.sample(set(np.where((ts_label_flag_1 == 0))[0]), num_indices*5)
    
    train_indices =  train_indices_1 + train_indices_0

    val_indices =  val_indices_1 + val_indices_0
    # total_train_val = set(train_indices + val_indices)
    # test_flag = (ts_l >= -1)
    # for ele in total_train_val:
    #     test_flag[ele] = 0
    
   
    # test_flag = (ts_l >= -1)
    # total_train_val = src_l[train_indices + val_indices]
    # for i in range(len(src_l)):
    #     if src_l[i] in total_train_val:
    #         test_flag[i] = 0
    
    # test_label_flag_1 = (test_flag) * (label_l)
    # test_label_flag_0 = (test_flag) * (1-label_l)
    # num_indices = 10
    # test_indices_1 = random.sample(set(np.where(test_label_flag_1 == 1)[0]), num_indices)
    # test_indices_0 = random.sample(set(np.where((test_label_flag_0 == 0))[0]), num_indices)
    # test_indices = test_indices_1 + test_indices_0
    

    # test_flag = 1 - task_time_p

    # test_indices = (1 - task_time_p)
    
    train_src_l = src_l[train_indices]
    train_dst_l = dst_l[train_indices]
    train_ts_l = ts_l[train_indices]
    train_e_idx_l = e_idx_l[train_indices]
    train_label_l = label_l[train_indices]


    val_src_l = src_l[val_indices]
    val_dst_l = dst_l[val_indices]
    val_ts_l = ts_l[val_indices]
    val_e_idx_l = e_idx_l[val_indices]
    val_label_l = label_l[val_indices]

    # use the true test dataset
    test_src_l = src_l[test_indices]

    test_dst_l = dst_l[test_indices]
    test_ts_l = ts_l[test_indices]
    test_e_idx_l = e_idx_l[test_indices]
    test_label_l = label_l[test_indices]

    
    # print(len(label_pool))
    # print(len(test_label_l))
    # for i in range(len(test_label_l)):
    #     if test_label_l[i]:
    #         count+=1
    # print("XXXXXXXX")
    # print(count)
    tgan = TGAN(full_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
# optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
# criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)
    tgan.load_state_dict(torch.load(model_path),strict=False)
    lr_model = LR(n_feat.shape[1])
    prompt = Tprog_prompt_layer(n_feat.shape[0],n_feat.shape[1])
    tgan.eval()
    lr_model = lr_model.train()

    prompt_optimizer = torch.optim.Adam(prompt.parameters(), lr=0.01)

 
    lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
    # time_prompt = time_prompt.to(device)
    prompt = prompt.to(device)
    lr_model = lr_model.to(device)
    lr_criterion = torch.nn.BCELoss()
    lr_criterion_eval = torch.nn.BCELoss()
    best_val_acc = 0
    best_val_auc = 0
    val_acc_re = []
    val_auc_re = []
    val_f1_re  = []
    for epoch in range(200):
        # indices_0 = random.sample(range(0, int(len(train_src_l)/2)),TRAIN_SHOT_NUM)
        # indices_1 = random.sample(range(int(len(train_src_l)/2),len(train_src_l)),TRAIN_SHOT_NUM)
        indices_1 = random.sample(range(0, 10),TRAIN_SHOT_NUM)
        indices_0 = random.sample(range(10,len(train_src_l)),TRAIN_SHOT_NUM*5)
        indices = indices_1 + indices_0
        src_l_cut = train_src_l[indices]
        dst_l_cut = train_dst_l[indices]
        ts_l_cut = train_ts_l[indices]
        label_l_cut = train_label_l[indices]
        prompt_optimizer.zero_grad()
  
        lr_optimizer.zero_grad()
        # with torch.no_grad():
        src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
        _,  _ ,src_t_ngh = full_ngh_finder.get_temporal_neighbor(src_l_cut, ts_l_cut, num_neighbors=1)
        src_t_ngh = src_t_ngh.reshape(1, len(src_t_ngh)) 
        delta_ts = ts_l_cut - src_t_ngh
        d_t = torch.from_numpy(delta_ts).float().to(device)
        # print(delta_ts)
        delta_ts_embed = tgan.time_encoder(d_t).squeeze(0)
        # print(delta_ts_embed.size())
        
        src_label = torch.from_numpy(label_l_cut).float().to(device)
        # embedding = prompt(src_embed)
        embedding = src_embed
    
        embedding = prompt(src_l_cut,delta_ts_embed,src_embed)
        
        lr_prob = lr_model(embedding).sigmoid()
        lr_loss = lr_criterion(lr_prob, src_label)
        lr_loss.backward()

        # for name, param in tgan.structure_prompt.named_parameters():
        #     print(f"{name} - Gradient: \n{param.grad}")
        lr_optimizer.step()
        prompt_optimizer.step()

        # val_src_l_cut = val_src_l[indices]
        # val_dst_l_cut = val_dst_l[indices]
        # val_ts_l_cut = val_ts_l[indices]
        # val_label_l_cut = val_label_l[indices]
        
        val_auc, val_acc, val_loss, val_f1 = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l, lr_model, tgan,prompt, VAL_SHOT_NUM)
        val_acc_re.append(val_acc)
        val_auc_re.append(val_auc)
        val_f1_re.append(val_f1)
        # if val_auc >  best_val_auc:
        #     best_val_auc = val_auc
        #     torch.save(lr_model.state_dict(), path_model)
        #     torch.save(prompt.state_dict(), path_prompt)
        #     torch.save(time_prompt.state_dict(), path_time_prompt)
        torch.save(lr_model.state_dict(), path_model)
        torch.save(prompt.state_dict(), path_prompt)
        torch.save(tgan.time_prompt.state_dict(), path_time_prompt)
        
    
    lr_model.load_state_dict(torch.load(path_model))
    prompt.load_state_dict(torch.load(path_prompt))
    tgan.time_prompt.load_state_dict(torch.load(path_time_prompt))
    
    TEST_SHOT_NUM = 0
    # print(len(test_src_l))
    test_auc, test_acc, test_loss,test_f1 = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, lr_model, tgan,prompt,TEST_SHOT_NUM)
    # logger.info(f'mean val acc: {sum(val_acc_re)/15},task:{task}')
 
    logger.info(f'test auc: {test_auc},task:{task}')
    logger.info(f'test f1: {test_f1},task:{task}')
    total_auc.append(test_auc)
    # total_acc.append(test_acc)
    total_f1.append(test_f1)

    #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))


np.savetxt(f"{DATA}_auc.txt", total_auc, fmt='%s')
np.savetxt(f"{DATA}_f1.txt", total_f1, fmt='%s')

# logger.info(f'wiki_total_mean_auc: {sum(total_auc)/TASK_NUM}')
# logger.info(f'wiki_total_mean_acc: {sum(total_acc)/TASK_NUM}')
# logger.info(f'wiki_total_mean_f1: {sum(total_f1)/TASK_NUM}')


# np.savetxt(f"{DATA}_total_mean_auc.txt", [sum(total_auc)/TASK_NUM], fmt='%s')
# np.savetxt(f"{DATA}_total_mean_f1.txt",[sum(total_f1)/TASK_NUM] ,fmt='%s')
NAME = args.name
# folder_path = "./Tprog/%s"%(DATA)  
FUNCTION = args.fn
folder_path = "./down_stream_result/%s/node/unsuper"%(FUNCTION) 


np.savetxt(f"{folder_path}/{NAME}_auc.txt", total_auc, fmt='%s')
np.savetxt(f"{folder_path}/{NAME}_f1.txt", total_f1, fmt='%s')

np.savetxt(f"{folder_path}/{NAME}_total_mean_auc.txt", [sum(total_auc)/TASK_NUM], fmt='%s')
np.savetxt(f"{folder_path}/{NAME}_total_mean_f1.txt",[sum(total_f1)/TASK_NUM] ,fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_total_mean_acc.txt",[sum(total_f1)/args.runs] ,fmt='%s')



 




