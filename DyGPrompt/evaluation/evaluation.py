import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors,batch_size=20):
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

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)
      # pos_prob = prompt(pos_prob).sigmoid()
      # neg_prob = prompt(neg_prob).sigmoid()
      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)
def eval_edge_prediction_fewshot(model, negative_edge_sampler, data, n_neighbors,batch_size=200):
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
    x= min(10, len(data.sources))
    indices =  np.random.choice(data.sources.size, 10, replace=False)

  
    sources_batch = data.sources[indices]
    destinations_batch = data.destinations[indices]
    timestamps_batch = data.timestamps[indices]
    edge_idxs_batch = data.edge_idxs[indices]

    size = len(sources_batch)
    _, negative_samples = negative_edge_sampler.sample(size)

    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                          negative_samples, timestamps_batch,
                                                          edge_idxs_batch, n_neighbors)
    # pos_prob = prompt(pos_prob).sigmoid()
    # neg_prob = prompt(neg_prob).sigmoid()
    pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    val_ap.append(average_precision_score(true_label, pred_score))
    val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)
import random
from sklearn.metrics import f1_score,precision_score
# def eval_node_classification(tgn, decoder, data, edge_idxs,shot_num, n_neighbors):
#   bs = None
#   with torch.no_grad():
#     decoder.eval()
#     tgn.eval()
#     if shot_num:
#       indices_1 = random.sample(range(0, 10),shot_num)
#       indices_0 = random.sample(range(10,len(data.sources)),shot_num*5)
#       indices = indices_1 + indices_0
#     else:
#       bs = 500
#     if not bs:
            
#       sources_batch = data.sources[indices]
#       destinations_batch = data.destinations[indices]
#       timestamps_batch = data.timestamps[indices]
#       edge_idxs_batch = edge_idxs[indices]
#       labels_batch = data.labels[indices]
#       source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,destinations_batch,destinations_batch,timestamps_batch,edge_idxs_batch,n_neighbors)
#       lr_prob = decoder(source_embedding).sigmoid()                                                                
#       pred_prob = lr_prob.cpu().numpy()
#       pred_label = pred_prob > 0.5
#       acc = (pred_label == labels_batch).mean()
#       f1 = f1_score(labels_batch, pred_label, average='binary')
#       auc_roc = roc_auc_score(labels_batch, pred_prob)
#       return  auc_roc, acc,f1
#     else:
#       pred_prob = np.zeros(len(data.sources))

#       num_instance = len(data.sources)
#       pred_prob = np.zeros(num_instance)
#       num_batch = math.ceil(num_instance / bs)
#       for k in range(num_batch):
#         s_idx = k * bs
#         e_idx = min(num_instance, s_idx + bs)

#         sources_batch = data.sources[s_idx: e_idx]
#         destinations_batch = data.destinations[s_idx: e_idx]
#         timestamps_batch = data.timestamps[s_idx:e_idx]
#         edge_idxs_batch = edge_idxs[s_idx: e_idx]
#         labels_batch = data.labels[s_idx:e_idx]
#         source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
#                                                                                     destinations_batch,
#                                                                                     destinations_batch,
#                                                                                     timestamps_batch,
#                                                                                     edge_idxs_batch,
#                                                                                     n_neighbors)
#         # src_label = torch.from_numpy(labels_batch).float()
#         lr_prob = decoder(source_embedding).sigmoid()    
#         pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
#       pred_label = pred_prob > 0.5
#       acc = (pred_label == data.labels).mean()
#       auc_roc = roc_auc_score(data.labels, pred_prob)
#       f1 = f1_score(data.labels, pred_label, average='binary')
#       return auc_roc,acc,f1
# def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
#   pred_prob = np.zeros(len(data.sources))
#   num_instance = len(data.sources)
#   num_batch = math.ceil(num_instance / batch_size)

#   with torch.no_grad():
#     decoder.eval()
#     tgn.eval()
#     for k in range(num_batch):
#       s_idx = k * batch_size
#       e_idx = min(num_instance, s_idx + batch_size)

#       sources_batch = data.sources[s_idx: e_idx]
#       destinations_batch = data.destinations[s_idx: e_idx]
#       timestamps_batch = data.timestamps[s_idx:e_idx]
#       edge_idxs_batch = edge_idxs[s_idx: e_idx]

#       source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
#                                                                                    destinations_batch,
#                                                                                    destinations_batch,
#                                                                                    timestamps_batch,
#                                                                                    edge_idxs_batch,
#                                                                                    n_neighbors)
#       pred_prob_batch = decoder(source_embedding).sigmoid()
#       pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

#   auc_roc = roc_auc_score(data.labels, pred_prob)
#   pred_label = pred_prob > 0.5
#   f1 = f1_score(data.labels, pred_label, average='binary')
#   acc = (pred_label == data.labels).mean()


#   return auc_roc,acc,f1
def eval_node_classification(tgn, decoder, data, edge_idxs,shot_num, n_neighbors):
  bs = None
  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    if shot_num:
      indices_1 = random.sample(range(0, 10),shot_num)
      indices_0 = random.sample(range(10,len(data.sources)),shot_num*5)
      indices = indices_1 + indices_0
    else:
      bs = 500
    if not bs:
            
      sources_batch = data.sources[indices]
      destinations_batch = data.destinations[indices]
      timestamps_batch = data.timestamps[indices]
      edge_idxs_batch = edge_idxs[indices]
      labels_batch = data.labels[indices]
      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,destinations_batch,destinations_batch,timestamps_batch,edge_idxs_batch,n_neighbors)
      
      lr_prob = decoder(source_embedding).sigmoid()                                                                
      pred_prob = lr_prob.cpu().numpy()
      pred_label = pred_prob > 0.5
      acc = (pred_label == labels_batch).mean()
      f1 = f1_score(labels_batch, pred_label, average='binary')
      auc_roc = roc_auc_score(labels_batch, pred_prob)
      return  auc_roc, acc,f1
    else:
      pred_prob = np.zeros(len(data.sources))

      num_instance = len(data.sources)
      pred_prob = np.zeros(num_instance)
      num_batch = math.ceil(num_instance / bs)
      
      for k in range(num_batch):
        s_idx = k * bs
        e_idx = min(num_instance, s_idx + bs)

        sources_batch = data.sources[s_idx: e_idx]
        destinations_batch = data.destinations[s_idx: e_idx]
        timestamps_batch = data.timestamps[s_idx:e_idx]
        edge_idxs_batch = edge_idxs[s_idx: e_idx]
        labels_batch = data.labels[s_idx:e_idx]
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                    destinations_batch,
                                                                                    destinations_batch,
                                                                                    timestamps_batch,
                                                                                    edge_idxs_batch,
                                                                                    n_neighbors)
        # src_label = torch.from_numpy(labels_batch).float()
        lr_prob = decoder(source_embedding).sigmoid()    
        pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
      pred_label = pred_prob > 0.5
      acc = (pred_label == data.labels).mean()
      auc_roc = roc_auc_score(data.labels, pred_prob)
      f1 = f1_score(data.labels, pred_label, average='binary')
      return auc_roc,acc,f1
def eval_node_classification_full(tgn, decoder, data, edge_idxs, batch_size, prompt,n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    prompt.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                             n_neighbors)
      source_embedding = prompt(source_embedding)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  pred_label = pred_prob > 0.5
  f1 = f1_score(data.labels, pred_label, average='binary')
  acc = (pred_label == data.labels).mean()


  return auc_roc,acc,f1
def eval_node_classification_GP(tgn, decoder, data, edge_idxs,shot_num,prompt, n_neighbors):
  bs = None
  with torch.no_grad():
    decoder.eval()
    prompt.eval()
    tgn.eval()
    if shot_num:
      indices_1 = random.sample(range(0, 10),shot_num)
      indices_0 = random.sample(range(10,len(data.sources)),shot_num*5)
      indices = indices_1 + indices_0
    else:
      bs = 500
    if not bs:
            
      sources_batch = data.sources[indices]
      destinations_batch = data.destinations[indices]
      timestamps_batch = data.timestamps[indices]
      edge_idxs_batch = edge_idxs[indices]
      labels_batch = data.labels[indices]
      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,destinations_batch,destinations_batch,timestamps_batch,edge_idxs_batch,n_neighbors)
      
      source_embedding = prompt(source_embedding)
      lr_prob = decoder(source_embedding).sigmoid()                                                                
      pred_prob = lr_prob.cpu().numpy()
      pred_label = pred_prob > 0.5
      acc = (pred_label == labels_batch).mean()
      f1 = f1_score(labels_batch, pred_label, average='binary')
      auc_roc = roc_auc_score(labels_batch, pred_prob)
      return  auc_roc, acc,f1
    else:
      pred_prob = np.zeros(len(data.sources))

      num_instance = len(data.sources)
      pred_prob = np.zeros(num_instance)
      num_batch = math.ceil(num_instance / bs)
      
      for k in range(num_batch):
        s_idx = k * bs
        e_idx = min(num_instance, s_idx + bs)

        sources_batch = data.sources[s_idx: e_idx]
        destinations_batch = data.destinations[s_idx: e_idx]
        timestamps_batch = data.timestamps[s_idx:e_idx]
        edge_idxs_batch = edge_idxs[s_idx: e_idx]
        labels_batch = data.labels[s_idx:e_idx]
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                    destinations_batch,
                                                                                    destinations_batch,
                                                                                    timestamps_batch,
                                                                                    edge_idxs_batch,
                                                                                    n_neighbors)
        # src_label = torch.from_numpy(labels_batch).float()
        source_embedding = prompt(source_embedding)
        lr_prob = decoder(source_embedding).sigmoid()    
        pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()
      pred_label = pred_prob > 0.5
      acc = (pred_label == data.labels).mean()
      auc_roc = roc_auc_score(data.labels, pred_prob)
      f1 = f1_score(data.labels, pred_label, average='binary')
      return auc_roc,acc,f1

