import torch
import rama_py
from collections import defaultdict
import numpy as np


opts = rama_py.multicut_solver_options("PD")
opts.sanitize_graph = True
opts.verbose = False

class CorrCluster():
    def __init__(self, use_node_score, oracle_edge, 
                 oracle_node, filter_edges, rank, min_samples):
        
        self.use_node_score = use_node_score
        self.oracle_edge = oracle_edge
        self.oracle_node = oracle_node
        self.filter_edges = filter_edges
        self.rank = rank
        self.min_samples = min_samples

    def corr_clustering(self,
            iter_data,
            edge_index,
            _node_score,
            _score,
            data,
            score,
            node_score,
            rama_cuda):
        i, (start, end) = iter_data
        # mask edges for sample in batch
        edge_mask = torch.logical_or(
            torch.logical_and(edge_index[0] >= start, edge_index[1] < end),
            torch.logical_and(edge_index[1] >= start, edge_index[0] < end))
        graph_edge_index = edge_index[:, edge_mask]
        src, dst = graph_edge_index

        # mask scores
        if self.use_node_score:
            graph_node_score = _node_score[start:end]
        graph_edge_score = _score[edge_mask]

        # if computcing oracle results
        if self.oracle_edge:
            graph_edge_score[data['point_instances'][src] == data['point_instances'][dst]] = 1
            graph_edge_score[data['point_instances'][src] != data['point_instances'][dst]] = 0
            graph_edge_score[data['point_instances'][src] <= 0] = 0
            graph_edge_score[data['point_instances'][dst] <= 0] = 0

            score[edge_mask] = graph_edge_score
            score[score == 0] = -10
            score[score == 1] = 10

        if self.oracle_node and self.use_node_score:
            graph_node_score[data['point_categories'][start:end]>0] = 1
            graph_node_score[data['point_categories'][start:end]<=0] = 0

            node_score[start:end] = graph_node_score
            node_score[node_score == 0] = -10
            node_score[node_score == 1] = 10

        # filter out edges with very low score already
        if self.filter_edges > 0:
            graph_edge_index = graph_edge_index[
                :, (graph_edge_score > self.filter_edges).squeeze()]
            graph_edge_score = graph_edge_score[
                (graph_edge_score > self.filter_edges).squeeze()]
        
        graph_edge_index = graph_edge_index - start.item()

        # filter out edges to/from nodes with low node score
        if self.use_node_score:
            graph_edge_score = graph_edge_score[torch.logical_and(
                graph_node_score[graph_edge_index[0]] > self.use_node_score, 
                graph_node_score[graph_edge_index[1]] > self.use_node_score).squeeze()]
            graph_edge_index = graph_edge_index[:, torch.logical_and(
                graph_node_score[graph_edge_index[0]] > self.use_node_score, 
                graph_node_score[graph_edge_index[1]] > self.use_node_score).squeeze()]

        # map nodes
        edges = torch.unique(graph_edge_index)
        mapping = torch.ones(end.item()-start.item()) * - 1
        mapping = mapping.int()
        mapping[edges] = torch.arange(edges.shape[0]).int()
        mapping = mapping.to(self.rank)
        _edge_index = graph_edge_index
        _edge_index[0, :] = mapping[graph_edge_index[0, :]]
        _edge_index[1, :] = mapping[graph_edge_index[1, :]]

        try:
            rama_out = rama_cuda(
                [e[0] for e in _edge_index.T.cpu().numpy()],
                [e[1] for e in _edge_index.T.cpu().numpy()], 
                (graph_edge_score.cpu().numpy()*2)-1,
                opts)
            mapped_clusters = torch.tensor(rama_out[0]).to(self.rank).int()
        except:
            mapped_clusters = torch.arange(edges.shape[0]).to(self.rank).int()

        # map back 
        _edge_index[0, :] = edges[_edge_index[0, :]]
        _edge_index[1, :] = edges[_edge_index[1, :]]
        clusters = torch.ones(end.item()-start.item()) * - 1
        clusters = clusters.int().to(self.rank)
        clusters[edges] = mapped_clusters

        # filter min_samples
        bin_count = torch.bincount(clusters)
        for i, count in enumerate(bin_count):
            if count < self.min_samples:
                clusters[clusters==i] = -1
            else:
                clusters[clusters==i] = i
        
        return clusters