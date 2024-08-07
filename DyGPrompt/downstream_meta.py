import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from model.tgn_ import TGN
from model.prompt import node_prompt_layer
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification_GP

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)
### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='hello', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=5100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--train_shot_num', type=int, default=3, help='')
parser.add_argument('--val_shot_num', type=int, default=3, help='')
parser.add_argument('--test_shot_num', type=int, default=100, help='')
parser.add_argument('--name', type=str, default='TGN WIKI ', help='Prefix to name the result txt')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--tag', type=int, default=3, help='')
try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)
TAG = args.tag
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
NAME = args.name
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
  node-classification.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
  node-classification.pth'

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

# full_data, node_features, edge_features, train_data, val_data, test_data = \
#   get_data_node_classification(DATA, use_validation=args.use_validation)

graph_df = pd.read_csv('./downstream_data/{}/ds_{}.csv'.format(DATA,DATA))
edge_features = np.load('./processed/ml_{}.npy'.format(DATA))
node_features = np.load('./processed/ml_{}_node.npy'.format(DATA))

  

val_time, test_time = list(np.quantile(graph_df.ts, [0.10, 0.20]))
sources = graph_df.u.values
destinations = graph_df.i.values
edge_idxs = graph_df.idx.values
labels = graph_df.label.values
timestamps = graph_df.ts.values

random.seed(2020)
label_flag = 0
task_start_time = 0
task_end_time = 0
for i in range(len(labels)):
  if labels[i]:
      label_flag += 1
#
  if label_flag ==20 and task_start_time == 0:
      task_start_time = timestamps[i]

  
  if label_flag ==24:
        task_end_time = timestamps[i]
        break
#
task_start = timestamps >= task_start_time
task_end =  timestamps <= task_end_time
task_time_p = task_start * task_end
task_time_pool = timestamps[task_time_p]
test_indices = (1 - task_time_p) > 0

test_indices = (1 - task_start) > 0
label_pool = labels[test_indices]
count = 0 
for i in range(len(labels)):
    if (timestamps[i]>task_end_time or timestamps[i]<task_start_time) and labels[i]:
        count +=1
#
task_time_set = random.sample(set(task_time_pool),100)
np.savetxt("wiki_task_time", task_time_set, fmt='%s')
  
  
  # train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  # test_mask = timestamps > test_time
  # val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

TRAIN_SHOT_NUM = args.train_shot_num
VAL_SHOT_NUM = args.val_shot_num
TEST_SHOT_NUM = args.test_shot_num
total_auc = []
total_acc = []
total_f1 = []
device = torch.device('cuda:{}'.format(GPU))
model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
from tqdm import tqdm
runs = args.n_runs
for task in tqdm(range(1)):
  #

  time_stamp = task_time_set[task]
  ts_flag = (timestamps <= time_stamp)
  index = np.where(timestamps == time_stamp)[0][0]
    
  #
  ts_label_flag_1 = (ts_flag) * (labels)

  ts_label_flag_1 = ts_label_flag_1[0:index+1]
  record = {}

  for i in range(len(ts_label_flag_1)-1,-1,-1):
      if sources[i] in record:
  
          # choose_node_flag[i] = 0
          ts_label_flag_1[i] = -1
      else:
          record[sources[i]] = 1 
  num_indices = 10  # 

  train_indices_1 = random.sample(set(np.where(ts_label_flag_1 == 1)[0]), num_indices)
  train_indices_0 = random.sample(set(np.where((ts_label_flag_1 == 0))[0]), num_indices*5)
  
  ts_label_flag_1[train_indices_1], ts_label_flag_1[train_indices_0] = -1, -2
  
  val_indices_1 = random.sample(set(np.where(ts_label_flag_1 == 1)[0]), num_indices)
  val_indices_0 = random.sample(set(np.where((ts_label_flag_1 == 0))[0]), num_indices*5)
  
  train_indices =  train_indices_1 + train_indices_0

  val_indices =  val_indices_1 + val_indices_0
  
  
  
  train_data = Data(sources[train_indices], destinations[train_indices], timestamps[train_indices],
                    edge_idxs[train_indices], labels[train_indices])
  val_data = Data(sources[val_indices], destinations[val_indices], timestamps[val_indices],
                  edge_idxs[val_indices], labels[val_indices])

  test_data = Data(sources[test_indices], destinations[test_indices], timestamps[test_indices],
                   edge_idxs[test_indices], labels[test_indices])




  max_idx = max(full_data.unique_nodes)

  train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

  # Set device
  device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_string)

  # Compute time statistics
  mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
  run_auc,run_acc,run_f1 = [],[],[]
  for i in range(runs):
    
    results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                  i) if i > 0 else "results/{}_node_classification.pkl".format(
      args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                edge_features=edge_features, device=device,
                n_layers=NUM_LAYER,
                n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                memory_update_at_start=not args.memory_update_at_end,
                embedding_module_type=args.embedding_module,
                message_function=args.message_function,
                aggregator_type=args.aggregator,
                memory_updater_type=args.memory_updater,
                n_neighbors=NUM_NEIGHBORS,
                mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                use_source_embedding_in_message=args.use_source_embedding_in_message,
                dyrep=args.dyrep,struc_prompt_tag=True,time_prompt_tag=True,meta_tag=True,tag=TAG)
    tgn = tgn.to(device)
    prompt = node_prompt_layer(edge_features.shape[1])

    model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
    print(model_path)
    tgn.load_state_dict(torch.load(model_path),strict=False)
    tgn.eval()
    logger.info('TGN models loaded')
    logger.info('Start training node classification task')

    decoder = MLP(node_features.shape[1], drop=DROP_OUT)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    decoder = decoder.to(device)
    prompt = prompt.to(device)
    
    struc_prompt_optimizer = torch.optim.Adam(tgn.struc_prompt.parameters(), lr=0.01)
    time_prompt_optimizer = torch.optim.Adam(tgn.time_prompt.parameters(), lr=0.01)
    meta_optimizer = torch.optim.Adam(tgn.meta_net.parameters(), lr=0.01)

    prompt_optimizer = torch.optim.Adam(prompt.parameters(), lr=0.01)
    decoder_loss_criterion = torch.nn.BCELoss()

    val_aucs = []
    train_losses = []

    # early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(args.n_epoch):
      # start_epoch = time.time()
      
      # Initialize memory of the model at each epoch
      if USE_MEMORY:
        tgn.memory.__init_memory__()
      indices_1 = random.sample(range(0, 10),TRAIN_SHOT_NUM)
      indices_0 = random.sample(range(10,len(train_data.sources)),TRAIN_SHOT_NUM*5)
      indices = indices_1 + indices_0

      tgn = tgn.eval()
      decoder = decoder.train()
      prompt.train()
      loss = 0
      
      sources_batch = train_data.sources[indices]
      destinations_batch = train_data.destinations[indices]
      timestamps_batch = train_data.timestamps[indices]
      edge_idxs_batch = full_data.edge_idxs[indices]
      labels_batch = train_data.labels[indices]

      size = len(sources_batch)

      decoder_optimizer.zero_grad()
      prompt_optimizer.zero_grad()
      #tag
      struc_prompt_optimizer.zero_grad()
      time_prompt_optimizer.zero_grad()
      meta_optimizer.zero_grad()
 
      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                  destinations_batch,
                                                                                  destinations_batch,
                                                                                  timestamps_batch,
                                                                                  edge_idxs_batch,
                                                                                  NUM_NEIGHBORS)

      labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
      source_embedding = prompt(source_embedding)
      pred = decoder(source_embedding).sigmoid()
      decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
      decoder_loss.backward()
      decoder_optimizer.step()
      prompt_optimizer.step()
      #tag
      struc_prompt_optimizer.step()
      time_prompt_optimizer.step()
      meta_optimizer.step()
      loss += decoder_loss.item()
      #
      train_losses.append(loss)

      val_auc = eval_node_classification_GP(tgn, decoder, val_data, full_data.edge_idxs,VAL_SHOT_NUM,prompt,
                                        n_neighbors=NUM_NEIGHBORS)
      val_aucs.append(val_auc)

      pickle.dump({
        "val_aps": val_aucs,
        "train_losses": train_losses,
        "epoch_times": [0.0],
        "new_nodes_val_aps": [],
      }, open(results_path, "wb"))
      torch.save(decoder.state_dict(), get_checkpoint_path(epoch))
    decoder.load_state_dict(torch.load(get_checkpoint_path(args.n_epoch-1)))
  
    decoder.eval()
    TEST_SHOT_NUM = 0
    test_auc,test_acc,test_f1 = eval_node_classification_GP(tgn, decoder, test_data, full_data.edge_idxs,TEST_SHOT_NUM,prompt,
                                          n_neighbors=NUM_NEIGHBORS)
    
    pickle.dump({
      "val_aps": val_aucs,
      "test_ap": test_auc,
      "train_losses": train_losses,
      "epoch_times": [0.0],
      "new_nodes_val_aps": [],
      "new_node_test_ap": 0,
    }, open(results_path, "wb"))
    run_auc.append(test_auc)
    run_acc.append(test_acc)
    run_f1.append(test_f1)

  total_auc.append(sum(run_auc)/runs)
  total_acc.append(sum(run_acc)/runs)
  total_f1.append(sum(run_f1)/runs)
  logger.info(f'task auc: {sum(run_auc)/runs}')
# folder_path = "./meta_result/%s"%(DATA)  
folder_path = "./"
# file_path = f"{folder_path}/{NAME}_f1.txt"  
# np.savetxt(f"{folder_path}/{NAME}_auc.txt", total_auc, fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_f1.txt", total_f1, fmt='%s')

np.savetxt(f"{folder_path}/{NAME}_total_mean_auc_f.txt", [sum(total_auc)/100], fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_total_mean_ff1.txt",[sum(total_f1)/100] ,fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_total_mean_acc_f.txt",[sum(total_acc)/100] ,fmt='%s')

  