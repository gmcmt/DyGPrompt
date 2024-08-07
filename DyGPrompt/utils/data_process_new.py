import numpy as np
import random
import pandas as pd


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


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./downstream_data/{}/ds_{}.csv'.format(dataset_name,dataset_name))
  edge_features = np.load('./processed/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./processed/ml_{}_node.npy'.format(dataset_name))

  

  val_time, test_time = list(np.quantile(graph_df.ts, [0.05, 0.1]))
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

  
  
  # train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  # test_mask = timestamps > test_time
  # val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
  #
  task_time_set = random.sample(set(task_time_pool),100)
  np.savetxt("wiki_task_time", task_time_set, fmt='%s')
  # train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
  #                   edge_idxs[train_mask], labels[train_mask])

  # val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
  #                 edge_idxs[val_mask], labels[val_mask])

  # test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
  #                  edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./processed/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./processed/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./processed/ml_{}_node.npy'.format(dataset_name)) 
    
  # if randomize_features:
  #   node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  test_time = list(np.quantile(graph_df.ts, [0.80]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  total_node_set = set(sources) | set(destinations)
  num_total_unique_nodes = len(total_node_set)

  # Compute nodes which appear at test time
  # test_node_set = set(sources[timestamps > val_time]).union(
  #   set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  # new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  # new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  # new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  # observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  # train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  mask_node_set = set(random.sample(set(sources[timestamps > test_time]).union(set(destinations[timestamps > test_time])), int(0.1 * num_total_unique_nodes)))

  mask_src_flag = graph_df.u.map(lambda x: x in mask_node_set).values
  mask_dst_flag = graph_df.i.map(lambda x: x in mask_node_set).values

  none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

  train_mask = (timestamps <= test_time) * (none_node_flag > 0)
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & mask_node_set) == 0
  new_node_set = total_node_set - train_node_set

  # val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  # test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    pass
    # n_new_nodes = len(new_test_node_set) // 2
    # val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    # test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    # edge_contains_new_val_node_mask = np.array(
    #   [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    # edge_contains_new_test_node_mask = np.array(
    #   [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    # new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    # new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    # new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    # new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # # validation and test with all edges
  # val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
  #                 edge_idxs[val_mask], labels[val_mask])

  # test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
  #                  edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  # new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
  #                          timestamps[new_node_val_mask],
  #                          edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  # new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
  #                           timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
  #                           labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  # print("The validation dataset has {} interactions, involving {} different nodes".format(
  #   val_data.n_interactions, val_data.n_unique_nodes))
  # print("The test dataset has {} interactions, involving {} different nodes".format(
  #   test_data.n_interactions, test_data.n_unique_nodes))
  # print("The new node validation dataset has {} interactions, involving {} different nodes".format(
  #   new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  # print("The new node test dataset has {} interactions, involving {} different nodes".format(
  #   new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  # print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
  #   len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
