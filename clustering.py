import yaml
import torch


class Clustering:
    def __init__(self, dataset_name, clustering_name, min_size=4, max_size=512, device='cpu'):
        with open(f'clusterings/{dataset_name}/{clustering_name}.yaml', 'r') as file:
            clusters = yaml.safe_load(file)

        num_nodes = max(idx for cluster in clusters for idx in cluster) + 1

        clusters = [cluster for cluster in clusters if min_size <= len(cluster) <= max_size]

        self.lengths = torch.tensor([len(cluster) for cluster in clusters])

        # Padding.
        max_len = max(len(cluster) for cluster in clusters)
        for cluster in clusters:
            cluster += [-1 for _ in range(max_len - len(cluster))]

        self.attn_mask = torch.zeros(len(clusters), 1, max_len, max_len, dtype=torch.bool, device=device)
        for i, length in enumerate(self.lengths):
            self.attn_mask[i, 0, :length, :length] = True

        self.forward_reshape_idx = torch.tensor(clusters, dtype=torch.int64, device=device)

        self.backward_reshape_double_idx = (
            torch.full(size=(num_nodes,), fill_value=-1, dtype=torch.int64, device=device),
            torch.full(size=(num_nodes,), fill_value=-1, dtype=torch.int64, device=device),
        )

        for i, cluster in enumerate(clusters):
            for j, node_id in enumerate(cluster):
                if node_id == -1:
                    break

                self.backward_reshape_double_idx[0][node_id] = i
                self.backward_reshape_double_idx[1][node_id] = j
