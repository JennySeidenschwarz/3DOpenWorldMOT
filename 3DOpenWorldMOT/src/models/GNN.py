import torch
from torch_geometric.nn.conv.message_passing import MessagePassing
import torch.nn as nn
import numpy as np
from collections import defaultdict
import rama_py
import random
import matplotlib
import os
import logging
import math
# import torchvision 
from .losses import sigmoid_focal_loss
from .GNN_utils import initial_edge_attributes, initial_node_attributes
from .correnlation_clustering import CorrCluster
from torch import multiprocessing as mp
import pickle

import torch.utils.checkpoint as checkpoint

rgb_colors = {}
for name, hex in matplotlib.colors.cnames.items():
    rgb_colors[name] = matplotlib.colors.to_rgb(hex)
rgb_colors = list(rgb_colors.values())
rgb_colors = rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
random.shuffle(rgb_colors)
rgb_colors[0] = (0, 0, 1)


logger = logging.getLogger("Model.GNN")


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class ClusterLayer(MessagePassing):
    def __init__(self, in_channel_node, in_channel_edge, out_channel_node, out_channel_edge, use_batchnorm=True, use_layernorm=False, use_drop=False, drop_p=0.4, skip_node_update=False):
        super().__init__(aggr='mean')
        # get edge mlp
        self.edge_mlp = torch.nn.Linear(in_channel_node * 2 + in_channel_edge, out_channel_edge)
        # get edge relu, bn, drop
        self.edge_relu = nn.ReLU(inplace=True)
        self.edge_batchnorm = nn.BatchNorm1d(out_channel_edge) \
            if use_batchnorm else use_batchnorm
        self.edge_layernorm = nn.LayerNorm(out_channel_edge) \
            if use_layernorm else use_layernorm
        self.edge_drop = nn.Dropout(p=drop_p) if use_drop else use_drop
        
        self.skip_node_update = skip_node_update
        if not self.skip_node_update:
            #get node mlp
            self.node_mlp = torch.nn.Linear(in_channel_node + out_channel_edge, out_channel_node)

            # get node relu, bn, drop
            self.node_relu = nn.ReLU(inplace=True)
            self.node_batchnorm = nn.BatchNorm1d(out_channel_node) \
                if use_batchnorm else use_batchnorm
            self.node_layernorm = nn.LayerNorm(out_channel_node) \
                if use_layernorm else use_layernorm
            self.node_drop = nn.Dropout(p=drop_p) if use_drop else use_drop

    def edge_updater(self, edge_attr, node_attr, edge_index):
        x1_i = node_attr[edge_index[0, :]]
        x1_i = x1_i.view(x1_i.shape[0], -1)
        # receiving
        x1_j = node_attr[edge_index[1, :]]
        x1_j = x1_j.view(x1_j.shape[0], -1)
        update_input = torch.cat([x1_i, x1_j, edge_attr], dim=1)

        edge_attr = self.edge_relu(self.edge_mlp(update_input))
        if self.edge_batchnorm:
            edge_attr = self.edge_batchnorm(edge_attr)
        if self.edge_layernorm:
            edge_attr = self.edge_layernorm(edge_attr)
        if self.edge_drop:
            edge_attr = self.edge_drop(edge_attr)
        return edge_attr

    def forward(self, node_attr, edge_index, edge_attr):
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        edge_attr = self.edge_updater(edge_attr, node_attr, edge_index)
        
        if not self.skip_node_update:
            # propagate_type: (x: OptPairTensor, alpha: Tensor)
            node_attr = self.propagate(edge_index, node_attr=node_attr, edge_attr=edge_attr)
        return node_attr, edge_index, edge_attr
    
    def propagate(self, edge_index, node_attr, edge_attr):
        dim_size = node_attr.shape[0]
        node_attr = self.message(node_attr[edge_index[0, :]], node_attr[edge_index[1, :]], edge_attr)
        node_attr = self.aggregate(node_attr, edge_index[1, :], dim_size=dim_size)
        return node_attr

    def message(self, x1_i, x1_j, edge_attr):
        # sending
        x1_i = x1_i.view(x1_i.shape[0], -1)
        # receiving
        x1_j = x1_j.view(x1_j.shape[0], -1)
        # only use sending and edge_attr
        tmp = torch.cat([x1_i, edge_attr], dim=1)
        x = self.node_relu(self.node_mlp(tmp))
        if self.node_batchnorm:
            x = self.node_batchnorm(x)
        if self.node_layernorm:
            x = self.node_layernorm(x)
        if self.node_drop:
            x = self.node_drop(x)
        return x

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str


class SevInpSequential(nn.Sequential):

    def __init__(self, gradient_checkpointing, layers):
        super().__init__(*layers)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                if self.gradient_checkpointing:
                    inputs = checkpoint.checkpoint(module, *inputs, use_reentrant=False)
                else:
                    inputs = module(*inputs)
            else:
                if self.gradient_checkpointing:
                    inputs = checkpoint.checkpoint(module, inputs, use_reentrant=False)
                else:
                    module(inputs)
        return inputs


class ClusterGNN(MessagePassing):
    def __init__(
            self,
            traj_channels,
            pos_channels,
            k=32,
            k_eval=64,
            r=0.5,
            graph='radius',
            edge_attr='diffpos',
            graph_construction='pos',
            node_attr='traj',
            cut_edges=0.5,
            min_samples=20,
            use_node_score=False,
            clustering='correlation',
            do_visualize=True,
            my_graph=True,
            oracle_node=False,
            oracle_edge=False,
            dataset='waymo',
            layer_sizes_edge=None,
            layer_sizes_node=None,
            ignore_stat_edges=0,
            ignore_stat_nodes=0,
            filter_edges=-1,
            node_loss=True,
            layer_norm=False,
            batch_norm=False,
            drop_out=False,
            augment=False,
            rank=0,
            gradient_checkpointing=False):
        super().__init__(aggr='mean')
        self.k = k
        self.k_eval = k_eval
        self.r = r
        self.graph = graph
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        self.graph_construction = graph_construction
        self.use_node_score = use_node_score * node_loss
        self.clustering = clustering
        if self.edge_attr == 'diffpos':
            edge_dim = pos_channels
        elif self.edge_attr == 'difftraj' or self.edge_attr == 'diffpostraj':
            edge_dim = traj_channels
        elif self.edge_attr == 'difftraj_diffpos':
            edge_dim = pos_channels + traj_channels
        elif self.edge_attr == 'pertime_diffpostraj':
            edge_dim = int(traj_channels/3)
        elif self.edge_attr == 'min_mean_max_diffpostrajtime' or self.edge_attr == 'min_mean_max_difftrajtime':
            edge_dim = 3
        elif self.edge_attr == 'min_mean_max_diffpostrajtime_normaldiff':
            edge_dim = 4
        
        # get node mlp
        self.node_attr = node_attr
        if self.node_attr == 'pos':
            node_dim = pos_channels
        elif self.node_attr == 'traj' or self.node_attr == 'postraj':
            node_dim = traj_channels
        elif self.node_attr == 'traj_pos':
            node_dim = traj_channels + pos_channels
        elif self.node_attr == 'min_mean_max_vel':
            node_dim = 3
        elif self.node_attr == 'min_mean_max_vel_normal':
            node_dim = 6

        layers = list()

        _node_dim = node_dim
        _edge_dim = edge_dim
        if layer_sizes_node is None:
            layer_sizes_node = {'l_1': node_dim}
            layer_sizes_edge = {'l_1': edge_dim}
        for j, (node_dim_hid, edge_dim_hid) in enumerate(zip(layer_sizes_node.values(), layer_sizes_edge.values())):
            if j == len(layer_sizes_node) -1 and not self.use_node_score:
                skip_node_update = True
            else:
                skip_node_update = False
            layers.append(ClusterLayer(
                in_channel_node=_node_dim,
                in_channel_edge=_edge_dim,
                out_channel_node=node_dim_hid,
                out_channel_edge=edge_dim_hid,
                use_batchnorm=batch_norm,
                use_layernorm=layer_norm,
                use_drop=drop_out,
                skip_node_update=skip_node_update))
            _node_dim = node_dim_hid
            _edge_dim = edge_dim_hid
        self.layers = SevInpSequential(gradient_checkpointing, layers)

        self.final = nn.Linear(_edge_dim, 1)
        if self.use_node_score:
            self.final_node = nn.Linear(_node_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # self.sigmoid = torch.nn.Tanh()
        self.cut_edges = cut_edges
        self.augment = augment
        self.min_samples = min_samples
        self.do_visualize = do_visualize
        self.my_graph = my_graph
        self.oracle_node = oracle_node
        self.oracle_edge = oracle_edge
        self.ignore_stat_edges = ignore_stat_edges
        self.ignore_stat_nodes = ignore_stat_nodes
        self.filter_edges = filter_edges
        self.dataset = dataset
        self.gradient_checkpointing = gradient_checkpointing
        self.CorrCluster = CorrCluster(use_node_score, oracle_edge, 
                 oracle_node, filter_edges, rank, min_samples)
        self.rank = rank

    def forward(self, data, eval=False, use_edge_att=True, name='General', corr_clustering=False):
        '''
        clustering: 'heuristic' / 'correlation'
        '''
        data = data.to(self.rank)
        batch_idx = data._slice_dict['pc_list']
        traj = data['traj']
        print(self.graph_construction)
        edge_index = data[self.graph_construction]

        # if no points in point cloud or no edges
        if traj.shape[0] == 0:
            return [None, None], list(), None, None
        elif edge_index.shape[1] == 0:
            return [None, None], torch.tensor(list(range(pc.shape[0]))), None, None
        
        # inputs to GNN
        traj = traj.view(traj.shape[0], -1)
        pc = data['pc_list']
        if 'pc_normals' in [k for k in data.keys]:
            point_normals = data['pc_normals']
        else:
            point_normals = None
        
        # add negative edges to edge_index
        if not eval and self.augment:
            edge_index = self.augment_graph()

        node_attr = initial_node_attributes(traj, pc, self.node_attr, point_normals, data['timestamps'], data['batch']).float()
        edge_attr = initial_edge_attributes(traj, pc, edge_index, point_normals).float()
        node_attr, edge_index, edge_attr = self.layers(node_attr, edge_index, edge_attr)

        # computes per edge index by:
        #   1. computing dot product between node features or
        #   2. directly uses edge attirbutes
        src, dst = edge_index
        if not use_edge_att:
            score = (node_attr[src] * node_attr[dst]).sum(dim=-1)
        else:
            score = self.final(edge_attr)
        
        # get final node score
        if self.use_node_score:
            node_score = self.final_node(node_attr)
        else:
            node_score = None

        # check for nans
        if torch.any(torch.isnan(score)):
            print('Having nan during forward pass...')
            return [torch.nan, torch.nan], edge_index, None

        # correlation clustering
        if eval and corr_clustering:
            # apply sigmoid to scores
            _score = self.sigmoid(score)
            if self.use_node_score:
                _node_score = self.sigmoid(node_score)
            else:
                _node_score = None

            data_loader = enumerate(zip(batch_idx[:-1], batch_idx[1:]))
            rama_cuda = rama_py.rama_cuda
            all_clusters = list()
            for iter_data in data_loader:
                clusters = CorrCluster.corr_clustering(
                    iter_data,
                    edge_index,
                    _node_score,
                    _score,
                    data,
                    score,
                    node_score,
                    rama_cuda)
                all_clusters.append(clusters)

            return [score, node_score], all_clusters, edge_index, None
        elif eval:
            return [score, node_score], [[]*len(batch_idx[:-1])], edge_index, None

        return [score, node_score], edge_index, None

    def augment_graph(self, data):
        point_instances = data.point_instances.unsqueeze(
            0) == data.point_instances.unsqueeze(0).T
        same_graph = data['batch'].unsqueeze(0) == data['batch'].unsqueeze(0).T
        point_instances = torch.logical_and(point_instances, same_graph)
        # setting edges that do not belong to object to zero
        point_instances[data.point_instances == 0, :] = False
        point_instances[:, data.point_instances == 0] = False
        num_pos = edge_index[:, point_instances[
            edge_index[0, :], edge_index[1, :]]].shape[1]
        num_neg = edge_index.shape[1] - num_pos

        # fast version
        missing_neg = int((num_pos - num_neg))
        a, b = list(), list()
        if missing_neg > 0 and point_instances.shape[0]:
            for _ in range(max(math.ceil(missing_neg/point_instances.shape[0])*2, 1)):
                a.append(torch.randperm(point_instances.shape[0]))
                b.append(torch.randperm(point_instances.shape[0]))
            a = torch.cat(a).to(self.rank)
            b = torch.cat(b).to(self.rank)
            a, b = a[~point_instances[a, b]], b[~point_instances[a, b]]
            a, b = a[:missing_neg], b[:missing_neg]
        elif point_instances.shape[0]:
            missing_pos = -missing_neg
            for _ in range(max(math.ceil(missing_pos/point_instances.shape[0])*2, 1)):
                a.append(torch.randperm(point_instances.shape[0]))
                b.append(torch.randperm(point_instances.shape[0]))
            a = torch.cat(a).to(self.rank)
            b = torch.cat(b).to(self.rank)
            a, b = a[point_instances[a, b]], b[point_instances[a, b]]
            a, b = a[:missing_pos], b[:missing_pos]
        add_idxs = torch.stack([a, b]).to(self.rank)
        edge_index = torch.cat([edge_index.T, add_idxs.T]).T

        return edge_index

    def visualize(self, nodes, edge_indices, pos, clusters, timestamp, mode='before', name='General'):
        os.makedirs(f'../../../vis_graph/{name}', exist_ok=True)
        import networkx as nx
        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        clusters_dict = defaultdict(list)
        for i, c in enumerate(clusters):
            clusters_dict[c].append(i)
        clusters = clusters_dict

        # adapt edges to predicted clusters
        if mode == 'after':
            edge_indices = list()
            for c, nodelist in clusters.items():
                if c == -1:
                    continue
                for i, node1 in enumerate(nodelist):
                    for j, node2 in enumerate(nodelist):
                        if j <= i:
                            continue
                        edge_indices.append([node1, node2])
            edge_indices = torch.tensor(edge_indices).to(self.rank).T

        # take only x and y position
        pos = pos[:, :-1]

        # make graph
        pos = {n.item(): p for n, p in zip(nodes, pos.cpu().numpy())}
        G = nx.Graph()
        G.add_nodes_from(nodes.numpy())
        G.add_edges_from(edge_indices.T.cpu().numpy())

        colors = [(0.999, 0.999, 0.999)] * nodes.shape[0]
        col_dict = dict()
        for i, (c, node_list) in enumerate(clusters.items()):
            for node in node_list:
                colors[node] = rgb_colors[i]
            col_dict[c] = rgb_colors[i]

        # save graph
        labels = {n.item(): str(n.item()) for n in nodes}
        plt.figure(figsize=(50, 50))
        nx.draw_networkx_edges(G, pos, width=3)
        nx.draw_networkx_nodes(G, pos, node_size=2, node_color=colors)
        # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_color='red')
        plt.axis("off")
        plt.savefig(f'../../../vis_graph/{name}/{timestamp}_{mode}.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


class GNNLoss(nn.Module):
    def __init__(
            self,
            bce_loss=False,
            node_loss=False,
            focal_loss_node=True,
            focal_loss_edge=True,
            alpha_node=0.25,
            alpha_edge=0.25,
            gamma_node=2,
            gamma_edge=2,
            rank=0,
            edge_weight=1,
            node_weight=1,
            ignore_stat_edges=0,
            ignore_stat_nodes=0) -> None:
        super().__init__()
        
        self.bce_loss = bce_loss
        self.node_loss = node_loss
        self.focal_loss_node = focal_loss_node
        self.focal_loss_edge = focal_loss_edge
        self.edge_weight = edge_weight
        self.node_weight = node_weight
        self.alpha_edge = alpha_edge
        self.alpha_node = alpha_node
        self.gamma_node = gamma_node
        self.gamma_edge = gamma_edge
        self.max_iter = 2000
        self.rank = rank
        self.ignore_stat_edges = ignore_stat_edges
        self.ignore_stat_nodes = ignore_stat_nodes
        self.sigmoid = torch.nn.Sigmoid()

        if not self.focal_loss_node:
            self._node_loss = nn.BCEWithLogitsLoss().to(self.rank)
        else:
            self._node_loss = sigmoid_focal_loss
        
        if not self.focal_loss_edge:
            self._edge_loss = nn.BCEWithLogitsLoss().to(self.rank)
        else:
            self._edge_loss = sigmoid_focal_loss
        
        self.test = nn.BCEWithLogitsLoss().to(self.rank)

    def forward(self, logits, data, edge_index, weight=False, weight_node=True, mode='train'):
        hist_node, hist_edge = None, None
        edge_logits, node_logits = logits
        loss = 0
        log_dict = dict()
        same_graph = data['batch'][edge_index[0, :]] == data['batch'][edge_index[1, :]]
        
        if self.bce_loss:
            point_instances = data.point_instances
            point_categories = data.point_categories[edge_index[1, :]]

            # get bool edge mask
            point_instances = point_instances[edge_index[0, :]] == point_instances[edge_index[1, :]]

            # keep only edges that belong to same graph (for batching opteration)
            point_instances = torch.logical_and(point_instances, same_graph)

            # setting edges that do not belong to object to zero
            # --> instance 0 is no object
            point_instances[data.point_instances[edge_index[0, :]] == 0] = False
            point_instances[data.point_instances[edge_index[1, :]] == 0] = False

            # sample edges
            point_instances = point_instances.to(self.rank)

            # if ignoring predictions for static edges in loss, get static edge filter
            if self.ignore_stat_edges:
                point_instances_stat = torch.logical_or(
                        torch.logical_and(
                            ~(data.point_instances_mov[edge_index[0, :]] != 0), 
                            data.point_instances[edge_index[0, :]] != 0),
                        torch.logical_and(
                            ~(data.point_instances_mov[edge_index[1, :]] != 0), 
                            data.point_instances[edge_index[1, :]] != 0))

                # filter edge logits, point instances and point categories
                edge_logits = edge_logits[~point_instances_stat]
                point_instances = point_instances[~point_instances_stat].float()
                point_categories = point_categories[~point_instances_stat]

            num_edge_pos, num_edge_neg = point_instances.sum(), (point_instances==0).sum()

            # compute loss
            if weight and not self.focal_loss_edge:
                # weight pos and neg samples
                num_pos = torch.sum((point_instances==1).float())
                num_neg = torch.sum((point_instances==0).float())
                pos_weight = num_neg/num_pos
                pos_weight = pos_weight.cpu()
                self._edge_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.rank)
            
            # compute loss
            if not self.focal_loss_edge:
                bce_loss_edge = self._edge_loss(
                    edge_logits.squeeze().float(), point_instances.squeeze().float())
            else:
                bce_loss_edge = self._edge_loss(
                    edge_logits.squeeze(),
                    point_instances.squeeze(),
                    alpha=self.alpha_edge,
                    gamma=self.gamma_edge,
                    reduction="mean",)
            
            # log loss
            loss += self.edge_weight * bce_loss_edge
            #print(f'{mode} bce loss edge', bce_loss_edge.detach().item())
            log_dict[f'{mode} bce loss edge'] = torch.zeros(2).to(self.rank)
            log_dict[f'{mode} bce loss edge'][0] = bce_loss_edge.detach().item()
            log_dict[f'{mode} bce loss edge'][1] = 1

            # get accuracy
            logits_rounded = self.sigmoid(edge_logits.clone().detach()).squeeze()
            hist_edge = np.histogram(logits_rounded.cpu().numpy(), bins=10, range=(0., 1.))
            logits_rounded[logits_rounded>0.5] = 1
            logits_rounded[logits_rounded<=0.5] = 0
            correct = logits_rounded == point_instances.squeeze()
            
            # overall
            log_dict[f'{mode} accuracy edge'] = torch.zeros(6).to(self.rank) 
            log_dict[f'{mode} accuracy edge'][0] = torch.sum(correct)/logits_rounded.shape[0]
            log_dict[f'{mode} accuracy edge'][1] = 1
            # negative edges
            if correct[point_instances==0].shape[0]:
                log_dict[f'{mode} accuracy edge'][2] = torch.sum(
                        correct[point_instances==0])/correct[point_instances==0].shape[0] 
                log_dict[f'{mode} accuracy edge'][3] = 1
            # positive edges
            if correct[point_instances==1].shape[0]:
                log_dict[f'{mode} accuracy edge'][4] = torch.sum(
                        correct[point_instances==1])/correct[point_instances==1].shape[0] 
                log_dict[f'{mode} accuracy edge'][5] = 1

            # per class accuracy:
            log_dict[f'{mode} accuracy edges connected to class'] = torch.zeros(40).to(self.rank)
            for c in torch.unique(point_categories):
                if correct[point_categories==c].shape[0]:
                    log_dict[f'{mode} accuracy edges connected to class'][2*c] = torch.sum(
                            correct[point_categories==c])/correct[point_categories==c].shape[0]
                    log_dict[f'{mode} accuracy edges connected to class'][2*c+1] = 1
            log_dict[f'{mode} num edge pos'] = num_edge_pos
            log_dict[f'{mode} num edge neg'] = num_edge_neg

        if self.node_loss:
            # get if point is object
            is_object = data.point_instances != 0
            is_object = is_object.type(torch.FloatTensor).to(self.rank)
            object_class = data.point_categories

            # if ignoring static nodes for loss computation get filter
            if self.ignore_stat_nodes:
                is_object_stat = torch.logical_and(
                    ~(data.point_instances_mov != 0), data.point_instances != 0)
                is_object_stat = is_object_stat.to(self.rank)

                # filter logits and object ground truth
                is_object = is_object[~is_object_stat]
                node_logits = node_logits[~is_object_stat]
                object_class = object_class[~is_object_stat]

            # weight pos and neg samples
            if weight_node and not self.focal_loss_node:
                num_pos = torch.sum((is_object==1).float())
                num_neg = torch.sum((is_object==0).float())
                pos_weight = num_neg/num_pos
                self._node_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.rank)
            
            # compute loss            
            if not self.focal_loss_node:
                bce_loss_node = self._node_loss(
                    node_logits.squeeze(), is_object.squeeze())
            else:
                bce_loss_node = self._node_loss(
                    node_logits.squeeze(),
                    is_object.squeeze(),
                    alpha=self.alpha_node,
                    gamma=self.gamma_node,
                    reduction="mean",)

            # log loss
            log_dict[f'{mode} bce loss node'] = torch.zeros(2).to(self.rank)
            log_dict[f'{mode} bce loss node'][0] = bce_loss_node.detach().item()
            log_dict[f'{mode} bce loss node'][1] = 1
            loss += self.node_weight * bce_loss_node

            # get accuracy
            logits_rounded_node = self.sigmoid(node_logits.clone().detach()).squeeze()
            hist_node = np.histogram(logits_rounded_node.cpu().numpy(), bins=10, range=(0., 1.))
            logits_rounded_node[logits_rounded_node>0.5] = 1
            logits_rounded_node[logits_rounded_node<=0.5] = 0
            correct = logits_rounded_node == is_object.squeeze()

            # Overall accuracy
            log_dict[f'{mode} accuracy node'] = torch.zeros(6).to(self.rank)
            log_dict[f'{mode} accuracy node'][0] = torch.sum(correct)/logits_rounded_node.shape[0]
            log_dict[f'{mode} accuracy node'][1] = 1
            # negative nodes
            if correct[is_object==0].shape[0]:
                log_dict[f'{mode} accuracy node'][2] = torch.sum(
                        correct[is_object==0])/correct[is_object==0].shape[0] 
                log_dict[f'{mode} accuracy node'][3] = 1
            # negative nodes
            if correct[is_object==1].shape[0]:
                log_dict[f'{mode} accuracy node'][4] = torch.sum(
                        correct[is_object==1])/correct[is_object==1].shape[0]
                log_dict[f'{mode} accuracy node'][5] = 1
            
            # per class
            log_dict[f'{mode} accuracy nodes of class'] = torch.zeros(40).to(self.rank)
            for c in torch.unique(object_class):
                if correct[object_class==c].shape[0]:
                    log_dict[f'{mode} accuracy nodes of class'][2*c] = torch.sum(
                            correct[object_class==c])/correct[object_class==c].shape[0]
                    log_dict[f'{mode} accuracy nodes of class'][2*c+1] = 1

            num_node_pos, num_node_neg = is_object.sum(), (is_object==0).sum()
            log_dict[f'{mode} num node pos'] = num_node_pos
            log_dict[f'{mode} num node neg'] = num_node_neg
        return loss, log_dict, hist_node, hist_edge
    
