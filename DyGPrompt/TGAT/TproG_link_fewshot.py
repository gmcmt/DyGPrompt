"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler
from prompt import Tprog_prompt_layer
### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
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
# NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim


MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_link/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

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


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label,prompt):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE=1024
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)
            # pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            src_embed,target_embed,background_embed = tgan.contrast_0(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            src_embed = Tprog(src_embed,src_l_cut,ts_l_cut,prompt,tgan)
            target_embed = Tprog(target_embed,dst_l_cut,ts_l_cut,prompt,tgan)
            background_embed = Tprog(background_embed,dst_l_fake,ts_l_cut,prompt,tgan)
            pos_score = tgan.affinity_score(src_embed, target_embed).squeeze(dim=-1)
            neg_score = tgan.affinity_score(src_embed, background_embed).squeeze(dim=-1)
            pos_prob = pos_score.sigmoid()
            neg_prob = neg_score.sigmoid()
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), val_f1, np.mean(val_auc)
def eval_one_epoch_0(hint, tgan, sampler, src, dst, ts, label,prompt):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
       
        tgan = tgan.eval()
        val_indices =  np.random.choice(src.size, 10, replace=False)
        TEST_BATCH_SIZE=512
    
           
       
        src_l_cut = src[val_indices]
        dst_l_cut = dst[val_indices]
        ts_l_cut = ts[val_indices]
        # label_l_cut = label[s_idx:e_idx]

        size = len(src_l_cut)
        src_l_fake, dst_l_fake = sampler.sample(size)
        # pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        src_embed,target_embed,background_embed = tgan.contrast_0(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        src_embed = Tprog(src_embed,src_l_cut,ts_l_cut,prompt,tgan)
        target_embed = Tprog(target_embed,dst_l_cut,ts_l_cut,prompt,tgan)
        background_embed = Tprog(background_embed,dst_l_fake,ts_l_cut,prompt,tgan)
        pos_score = tgan.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = tgan.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        pos_prob = pos_score.sigmoid()
        neg_prob = neg_score.sigmoid()
        pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
        pred_label = pred_score > 0.5
        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        
        val_acc.append((pred_label == true_label).mean())
        val_ap.append(average_precision_score(true_label, pred_score))
        # val_f1.append(f1_score(true_label, pred_label))
        val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), val_f1, np.mean(val_auc)
### Load data and train val test split
g_df = pd.read_csv('./downstream_data/{}/ds_{}.csv'.format(DATA,DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [0.10, 0.20]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

random.seed(2020)
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

# validation and test with all edges
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]
# validation and test with edges that at least has one new node (not in training set)
nn_val_src_l = src_l[nn_val_flag]
nn_val_dst_l = dst_l[nn_val_flag]
nn_val_ts_l = ts_l[nn_val_flag]
nn_val_e_idx_l = e_idx_l[nn_val_flag]
nn_val_label_l = label_l[nn_val_flag]

nn_test_src_l = src_l[nn_test_flag]
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag]
nn_test_label_l = label_l[nn_test_flag]

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.affinity_score.parameters(), lr=LEARNING_RATE)
# tgan.load_state_dict(torch.load(model_path),strict=False)
time_prompt_optimizer = torch.optim.Adam(tgan.time_prompt.parameters(), lr=0.01)
meta_prompt_optimizer = torch.optim.Adam(tgan._meta_net.parameters(), lr=1)
meta_prompt_optimizer_1 = torch.optim.Adam(tgan.meta_net_t.parameters(), lr=0.001)
structure_prompt_optimizer = torch.optim.Adam(tgan.structure_prompt.parameters(),lr=0.01)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

early_stopper = EarlyStopMonitor()
test_aps = []
test_aucs = []
test_nn_aps = []
test_nn_aucs = []
def Tprog(src_embed,src_l_cut, ts_l_cut,prompt,model,ngh_finder = full_ngh_finder,device=device):
    _,  _ ,src_t_ngh = full_ngh_finder.get_temporal_neighbor(src_l_cut, ts_l_cut, num_neighbors=1)
    src_t_ngh = src_t_ngh.reshape(1, len(src_t_ngh)) 
    delta_ts = ts_l_cut - src_t_ngh
    d_t = torch.from_numpy(delta_ts).float().to(device)
    delta_ts_embed = model.time_encoder(d_t).squeeze(0)

    embedding = prompt(src_l_cut,delta_ts_embed,src_embed)
    return embedding
tgan.load_state_dict(torch.load(model_path),strict=False)

for task in range(100):
    logger.info('start {} task'.format(task))
    prompt = Tprog_prompt_layer(n_feat.shape[0],n_feat.shape[1]) 
    prompt_optimizer = torch.optim.Adam(prompt.parameters(), lr=0.01)
    prompt = prompt.to(device)


    for epoch in range(50):
        # Training 
        # training use only training graph
        tgan.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        
        train_indices =  np.random.choice(train_src_l.size, 10, replace=False)
       
        src_l_cut, dst_l_cut = train_src_l[train_indices], train_dst_l[train_indices]
        ts_l_cut = train_ts_l[train_indices]
        label_l_cut = train_label_l[train_indices]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            
        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)
            
        optimizer.zero_grad()
        prompt_optimizer.zero_grad()
        # meta_prompt_optimizer.zero_grad()
        # meta_prompt_optimizer_1.zero_grad()
        # time_prompt_optimizer.zero_grad()
        # structure_prompt_optimizer.zero_grad()
            
        tgan = tgan.train()
        # pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        src_embed,target_embed,background_embed = tgan.contrast_0(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        src_embed = Tprog(src_embed,src_l_cut,ts_l_cut,prompt,tgan)
        target_embed = Tprog(target_embed,dst_l_cut,ts_l_cut,prompt,tgan)
        background_embed = Tprog(background_embed,dst_l_fake,ts_l_cut,prompt,tgan)
        pos_score = tgan.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = tgan.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        # print(neg_score.size())
        # print(neg_score.size())
        pos_prob = pos_score.sigmoid()
        neg_prob = neg_score.sigmoid()
        
        # meta_prompt_optimizer.step()
        # meta_prompt_optimizer_1.step()
        # time_prompt_optimizer.step()
        # structure_prompt_optimizer.step()
            # get training results
     
        tgan = tgan.eval()
        pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
        pred_label = pred_score > 0.5

        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        # acc.append((pred_label == true_label).mean())
        # ap.append(average_precision_score(true_label, pred_score))
        # print(pos_prob.size())
        # print(pos_label.size())
        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)
        
        loss.backward()
        optimizer.step()
        prompt_optimizer.step()
        # f1.append(f1_score(true_label, pred_label))
        m_loss.append(loss.item())
   

        # validation phase use all information
        # tgan.ngh_finder = full_ngh_finder
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch_0('val for old nodes', tgan, val_rand_sampler, val_src_l, 
        val_dst_l, val_ts_l, val_label_l,prompt)

        # nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch_0('val for new nodes', tgan, nn_val_rand_sampler, nn_val_src_l, 
        # nn_val_dst_l, nn_val_ts_l, nn_val_label_l)
        
            
        # logger.info('epoch: {}:'.format(epoch))
        # logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        # logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
        # logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
        # logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
        # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
        test_dst_l, test_ts_l, test_label_l,prompt)


    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
    nn_test_dst_l, nn_test_ts_l, nn_test_label_l,prompt)
    test_aps.append(test_ap)
    test_aucs.append(test_auc)
    print(test_auc)
    test_nn_aps.append(nn_test_ap)
    test_nn_aucs.append(nn_test_auc)
        # if early_stopper.early_stop_check(val_ap):
        #     logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        #     logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        #     best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        #     tgan.load_state_dict(torch.load(best_model_path))
        #     logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        #     tgan.eval()
        #     break
        # else:
        #     # torch.save(tgan.state_dict(), get_checkpoint_path(epoch))
        #     pass
        


# testing phase use all information
NAME = args.name
FUNCTION = args.fn
folder_path = "./down_stream_result/%s/link/unsuper"%(FUNCTION)  
tgan.ngh_finder = full_ngh_finder


# x=[]
# y = []



np.savetxt(f"{folder_path}/{NAME}_aps.txt", [sum(test_aps)/100], fmt='%s')


np.savetxt(f"{folder_path}/{NAME}_aucs.txt", [sum(test_aucs)/100], fmt='%s')

np.savetxt(f"{folder_path}/{NAME}_nn_aps.txt", [sum(test_nn_aps)/100], fmt='%s')


np.savetxt(f"{folder_path}/{NAME}_nn_aucs.txt", [sum(test_nn_aucs)/100], fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_total_mean_f1.txt",[sum(total_f1)/TASK_NUM] ,fmt='%s')




# logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
# logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

# logger.info('Saving TGAN model')
# torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
# logger.info('TGAN models saved')

 



