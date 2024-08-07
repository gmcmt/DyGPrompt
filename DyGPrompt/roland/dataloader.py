from torch.utils.data import Dataset, DataLoader
import torch
class SnapshotNodeDataset(Dataset):
    def __init__(self, node_features, source_indices, dest_indices):
        self.node_features = node_features
        self.source_indices = source_indices
        self.dest_indices = dest_indices

    def __len__(self):
  
        return len(self.source_indices)

    def __getitem__(self, idx):

        source_idx = self.source_indices[idx]
        dest_idx = self.dest_indices[idx]

        source_feature = self.node_features.weight[source_idx]
        dest_feature = self.node_features.weight[dest_idx]

        return source_feature, dest_feature, source_idx, dest_idx
