import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from model.prompt import Tprog_prompt_layer
from evaluation.evaluation import eval_edge_prediction_fewshot,eval_edge_prediction
from model.tgn_ import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_d_data, compute_time_statistics
from sklearn.metrics import average_precision_score, roc_auc_score
torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=5, help='Number of epochs')
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
parser.add_argument('--train_shot_num', type=int, default=3)
parser.add_argument('--val_shot_num', type=int, default=3)
parser.add_argument('--test_shot_num', type=int, default=100)
parser.add_argument('--name', type=str, default='', help='Prefix to name the result txt')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--tag', type=int, default=1)
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
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}-{"prompt"}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
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

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_d_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
from model.prompt import node_prompt_layer

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
# total_aps = []
# total_auc = []
test_aps = []
test_aucs = [] 
test_nn_aps = []
test_nn_aucs = [] 

def Tprog(src_embed,src_l_cut, ts_l_cut,prompt,model,ngh_finder = full_ngh_finder,device=device):
    _,  _ ,src_t_ngh = full_ngh_finder.get_temporal_neighbor(src_l_cut, ts_l_cut, n_neighbors=1)
    src_t_ngh = src_t_ngh.reshape(1, len(src_t_ngh)) 
    delta_ts = ts_l_cut - src_t_ngh
    d_t = torch.from_numpy(delta_ts).float().squeeze(0).to(device)

    # delta_ts_embed = model.time_encoder(d_t).squeeze(0)
    delta_ts_embed = model.time_encoder(d_t.unsqueeze(dim=1)).view(len(
      src_l_cut), -1)
    embedding = prompt(src_l_cut,delta_ts_embed,src_embed)
    return embedding
def eval_edge_prediction_TP(model, negative_edge_sampler, data, n_neighbors,batch_size=1024):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      source_node_embedding, destination_node_embedding, negative_node_embedding = model.compute_link_probabilities(sources_batch, destinations_batch, negative_samples,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
        


      source_node_embedding = Tprog(source_node_embedding,sources_batch,timestamps_batch,prompt,model)
      destination_node_embedding = Tprog(destination_node_embedding,destinations_batch,timestamps_batch,prompt,model)
      negative_node_embedding = Tprog(negative_node_embedding,negative_samples,timestamps_batch,prompt,model)
      score = tgn.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
      n_samples = len(sources_batch)
      pos_score = score[:n_samples]
      neg_score = score[n_samples:]

      pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()
    #   pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
    #                                                         negative_samples, timestamps_batch,
    #                                                         edge_idxs_batch, n_neighbors)
      # pos_prob = prompt(pos_prob).sigmoid()
      # neg_prob = prompt(neg_prob).sigmoid()
      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)
prompt = Tprog_prompt_layer(node_features.shape[0],node_features.shape[1])
prompt_optimizer = torch.optim.Adam(prompt.parameters(), lr=0.01)
prompt = prompt.to(device)
for task in range(100):
    i = 0
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)
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
                dyrep=args.dyrep,struc_prompt_tag=False, time_prompt_tag=False, meta_tag=False, tag=1)
    criterion = torch.nn.BCELoss()
    model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
    tgn.load_state_dict(torch.load(model_path),strict=False)


    #optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)

    # prompt_optimizer = torch.optim.Adam(tgn.prompt.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(tgn.affinity_score.parameters(), lr=LEARNING_RATE)
    
    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))

        loss = 0
        #   optimizer.zero_grad()
        prompt_optimizer.zero_grad()
        optimizer.zero_grad()
  

        x= min(10, len(train_data.sources))

        train_indices =  np.random.choice(train_data.sources.size, 10, replace=False)


        sources_batch, destinations_batch = train_data.sources[train_indices], \
                                            train_data.destinations[train_indices]
        edge_idxs_batch = train_data.edge_idxs[train_indices]
        timestamps_batch = train_data.timestamps[train_indices]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        source_node_embedding, destination_node_embedding, negative_node_embedding = tgn.compute_link_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
        


        source_node_embedding = Tprog(source_node_embedding,sources_batch,timestamps_batch,prompt,tgn)
        destination_node_embedding = Tprog(destination_node_embedding,destinations_batch,timestamps_batch,prompt,tgn)
        negative_node_embedding = Tprog(negative_node_embedding,negatives_batch,timestamps_batch,prompt,tgn)
        score = tgn.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
        n_samples = len(sources_batch)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        pos_prob, neg_prob = pos_score.sigmoid(), neg_score.sigmoid()
        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

        loss /= args.backprop_every

        loss.backward()
        optimizer.step()
        prompt_optimizer.step()
   
        # prompt_optimizer.step()
        m_loss.append(loss.item())

        # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
        # the start of time
        if USE_MEMORY:
            tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
            train_memory_backup = tgn.memory.backup_memory()

        # val_ap, val_auc = eval_edge_prediction_fewshot(model=tgn,
        #                                                         negative_edge_sampler=val_rand_sampler,
        #                                                         data=val_data,
        #                                                         n_neighbors=NUM_NEIGHBORS)
        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()
        # Restore memory we had at the end of training to be used when validating on new nodes.
        # Also backup memory after validation so it can be used for testing (since test edges are
        # strictly later in time than validation edges)
            tgn.memory.restore_memory(train_memory_backup)

        # Validate on unseen nodes
        nn_val_ap, nn_val_auc = eval_edge_prediction_TP(model=tgn,
                                                                            negative_edge_sampler=val_rand_sampler,
                                                                            data=new_node_val_data,
                                                                            n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
        # Restore memory we had at the end of validation
            tgn.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        # val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        # pickle.dump({
        # "val_aps": val_aps,
        # "new_nodes_val_aps": new_nodes_val_aps,
        # "train_losses": train_losses,
        # "epoch_times": epoch_times,
        # "total_epoch_times": total_epoch_times
        # }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        # logger.info(
        # 'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        # logger.info(
        # 'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

        # Early stopping
        # if early_stopper.early_stop_check(val_ap):
        #     logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        #     logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        #     best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        #     tgn.prompt.load_state_dict(torch.load(best_model_path))
        #     logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        #     tgn.eval()
        #     # break
        # else:
        #     torch.save(tgn.prompt.state_dict(), get_checkpoint_path(epoch))

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    ### Test
    tgn.embedding_module.neighbor_finder = full_ngh_finder
    test_ap, test_auc = eval_edge_prediction_TP(model=tgn,
                                                                negative_edge_sampler=test_rand_sampler,
                                                                data=test_data,
                                                                n_neighbors=NUM_NEIGHBORS)
    test_aps.append(test_ap)
    test_aucs.append(test_auc)
    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    # Test on unseen nodes
    nn_test_ap, nn_test_auc = eval_edge_prediction_TP(model=tgn,
                                                                            negative_edge_sampler=nn_test_rand_sampler,
                                                                            data=new_node_test_data,
                                                                            n_neighbors=NUM_NEIGHBORS)
    test_nn_aps.append(nn_test_ap)
    test_nn_aucs.append(nn_test_auc)

    logger.info(
        'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
    logger.info(
        'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
    # Save results for this run
    # pickle.dump({
    #     "val_aps": val_aps,
    #     "new_nodes_val_aps": new_nodes_val_aps,
    #     "test_ap": test_ap,
    #     "new_node_test_ap": nn_test_ap,
    #     "epoch_times": epoch_times,
    #     "train_losses": train_losses,
    #     "total_epoch_times": total_epoch_times
    # }, open(results_path, "wb"))

    #   logger.info('Saving TGN model')
    #   if USE_MEMORY:
    #     # Restore memory at the end of validation (save a model which is ready for testing)
    #     tgn.memory.restore_memory(val_memory_backup)
    #   torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    #   logger.info('TGN model saved')

folder_path = "./link_result/%s/fewshot"%(DATA)  

np.savetxt(f"{folder_path}/{NAME}_auc.txt", test_aucs, fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_ap.txt", test_aps, fmt='%s')

np.savetxt(f"{folder_path}/{NAME}_total_mean_aucs.txt", [sum(test_aucs)/100], fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}_total_mean_ap.txt",[sum(test_aps)/100] ,fmt='%s')

np.savetxt(f"{folder_path}/{NAME}nn_total_mean_aucs.txt", [sum(test_nn_aucs)/100], fmt='%s')
# np.savetxt(f"{folder_path}/{NAME}nn_total_mean_ap.txt",[sum(test_nn_aps)/100] ,fmt='%s')
