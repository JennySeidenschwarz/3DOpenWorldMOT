import torch
import torch.nn as nn
import sklearn.metrics
from torch_geometric.nn import knn_graph, radius_graph


def simplediff(a, b):
    return b-a


def initial_edge_attributes(x1, x2, edge_index, _type, point_normals=None, distance='euclidean'):
    # Given edge-level attention coefficients for source and target nodes,
    # we simply need to sum them up to "emulate" concatenation:
    if _type == 'diffpos':
        a = x2[edge_index[0]]
        b = x2[edge_index[1]]
        d = simplediff
    elif _type == 'difftraj':
        a = x1[edge_index[0]]
        b = x1[edge_index[1]]
        d = simplediff
    elif _type == 'difftraj_diffpos':
        a = torch.stack([x2[edge_index[0]], x1[edge_index[0]]])
        b = torch.stack([x2[edge_index[1]], x1[edge_index[1]]])
        d = simplediff
    elif _type == 'diffpostraj':
        a = x2[edge_index[0]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[0]]
        b = x2[edge_index[1]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[1]]
        d = simplediff
    elif _type == 'pertime_difftraj' or _type == 'min_mean_max_difftrajtime':
        a = x1.view(x1.shape[0], -1, 3)[edge_index[0]]
        a = a.view(edge_index.shape[1], -1)

        b = x1.view(x1.shape[0], -1, 3)[edge_index[1]]
        b = b.view(edge_index.shape[1], -1)

        a_shape = a.shape
        a = a.view(-1, 3)
        b = b.view(-1, 3)
        d = torch.nn.PairwiseDistance(p=2)

    elif _type == 'pertime_diffpostraj' or _type == 'min_mean_max_diffpostrajtime' or \
            _type == 'min_mean_max_diffpostrajtime_normaldiff':
        a = x1.view(x1.shape[0], -1, 3)[edge_index[0]]+x2[edge_index[0]].unsqueeze(1)
        a = a.view(edge_index.shape[1], -1)

        b = x1.view(x1.shape[0], -1, 3)[edge_index[1]]+x2[edge_index[1]].unsqueeze(1)
        b = b.view(edge_index.shape[1], -1)

        a_shape = a.shape
        a = a.view(-1, 3)
        b = b.view(-1, 3)
        d = torch.nn.PairwiseDistance(p=2)

        if _type == 'min_mean_max_diffpostrajtime_normaldiff':
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_normals = 1 - cos(point_normals[edge_index[0]], point_normals[edge_index[1]])
    
    edge_attr = d(a, b)
    if _type == 'pertime_diffpostraj':
        edge_attr = edge_attr.view(a_shape[0], -1)
    elif _type == 'min_mean_max_diffpostrajtime' or _type == 'min_mean_max_difftrajtime':
        edge_attr = edge_attr.view(a_shape[0], -1)
        edge_attr = torch.vstack([
            edge_attr.min(dim=-1).values,
            edge_attr.max(dim=-1).values,
            edge_attr.mean(dim=-1)]).T
    elif _type == 'min_mean_max_diffpostrajtime_normaldiff':
        edge_attr = edge_attr.view(a_shape[0], -1)
        edge_attr = torch.vstack([
            edge_attr.min(dim=-1).values,
            edge_attr.max(dim=-1).values,
            edge_attr.mean(dim=-1),
            cos_normals]).T

    return edge_attr


def initial_node_attributes(x1, x2, _type, point_normals=None, timestamps=None, batch=None, dataset='waymo'):
    # Given edge-level attention coefficients for source and target nodes,
    # we simply need to sum them up to "emulate" concatenation:
    if _type == 'pos':
        node_attr = x2
    elif _type == 'traj':
        node_attr = x1
    elif _type == 'traj_pos':
        node_attr = torch.stack([x2, x1])
    elif _type == 'postraj' or _type =='mean_dist_over_time':
        node_attr = x1.view(x1.shape[0], -1, 3)+x2.unsqueeze(1)
        if _type == 'postraj':
            node_attr = node_attr.view(node_attr.shape[0], -1)
    elif _type == 'min_mean_max_vel' or 'min_mean_max_vel_normal':
        time = timestamps[batch, :]
        node_attr = x1.view(x1.shape[0], -1, 3)
        diff_time = timestamps[batch, 1:] - timestamps[batch, :-1]
        if 'Argo' in dataset:
            diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
        else:
            diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)
        node_attr = node_attr[:, 1:, :] - node_attr[:, :-1, :]
        node_attr = torch.linalg.norm(node_attr, dim=-1)
        node_attr = node_attr / diff_time                
        node_attr = torch.vstack([
            node_attr.min(dim=-1).values,
            node_attr.max(dim=-1).values,
            node_attr.mean(dim=-1)]).T
        if _type == 'min_mean_max_vel_normal':
            node_attr = torch.hstack([node_attr, point_normals])

    return node_attr


def get_graph(k, r, _type, graph_attr, my_graph):
    # get edges using knn graph (for computational feasibility)
    if _type == 'knn':
        if my_graph and len(graph_attr.shape) != 2:
            edge_index = _my_graph(
                graph_attr, r, max_num_neighbors=k, type='knn')
        else:
            edge_index = knn_graph(x=graph_attr, k=k)
    elif _type == 'radius':
        if my_graph and len(graph_attr.shape) != 2:
            edge_index = _my_graph(
                graph_attr, r, max_num_neighbors=k, type='radius')
        else:
            edge_index = radius_graph(graph_attr, r, max_num_neighbors=k)
    
    return edge_index


def _my_graph(X, r=5, max_num_neighbors=16, type='radius', metric='euclidean', graph_construction='pos', rank=0):
    # get distances between nodes
    if graph_construction == 'pos' or graph_construction == 'min_mean_max_vel':
        dist = torch.from_numpy(sklearn.metrics.pairwise_distances(X.cpu().numpy(), metric=metric)).to(self.rank)
    else:                
        # following two lines are faster but cuda oom
        # dist = torch.cdist(X_time, X_time)
        # dist = dist.mean(dim=0)
        dist = torch.zeros(X.shape[0], X.shape[0]).to(rank)
        for t in range(X.shape[1]):
            dist += torch.cdist(X[:, t, :].unsqueeze(0),X[:, t, :].unsqueeze(0)).squeeze()
        dist = dist / X.shape[1]

    # set diagonal elements to 0to have no self-loops
    dist.fill_diagonal_(100)

    # get indices up to max_num_neighbors per node --> knn neighbors
    num_neighbors = min(max_num_neighbors, dist.shape[0])
    idxs_0 = torch.tile(torch.arange(dist.shape[0]).unsqueeze(1).to(rank), (1, num_neighbors)).flatten()
    idxs_1 = dist.topk(k=num_neighbors, dim=1, largest=False).indices.flatten()

    # if radius graph, filter nodes that are within radius 
    # but don't exceed max num neighbors
    if type == 'radius':
        dist = dist[idxs_0, idxs_1]
        idx = torch.where(dist<r)[0]
        idxs_0, idxs_1 = idxs_0[idx], idxs_1[idx]

    edge_index = torch.vstack([idxs_0, idxs_1])

    return edge_index