from collections import defaultdict
from turtle import forward
from sklearn import cluster
import torch
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn import knn_graph, radius_graph
import torch.nn.functional as F
from torch_geometric.utils import softmax
import torch.nn as nn
import numpy as np
import matplotlib
import random

rgb_colors = {}
for name, hex in matplotlib.colors.cnames.items():
    rgb_colors[name] = matplotlib.colors.to_rgb(hex)
rgb_colors = list(rgb_colors.values())
random.shuffle(rgb_colors)


class SimpleGraph(MessagePassing):
    def __init__(self, k=32, r=0.5, graph='radius', edge_attr='diffpos', node_attr='diffpos', cut_edges=0.2, min_samples=5):
        super().__init__(aggr='mean')
        self.k = k
        self.r = r
        self.graph = graph
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        self.cut_edges = cut_edges
        self.min_samples = min_samples

    def initial_edge_attributes(self, x1, x2, edge_index):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.edge_attr == 'diffpos':
            edge_attr = x2[edge_index[0]] - x2[edge_index[1]]
        elif self.edge_attr == 'difftraj':
            edge_attr = x1[edge_index[0]] - x1[edge_index[1]]
        elif self.edge_attr == 'difftraj_diffpos':
            edge_attr = torch.hstack([x1[edge_index[0]] - x1[edge_index[1]], x2[edge_index[0]] - x2[edge_index[1]]])

        return edge_attr
    
    def initial_node_attributes(self, x1, x2):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.node_attr == 'pos':
            node_attr = x2
        elif self.node_attr == 'traj':
            node_attr = x1
        elif self.node_attr == 'traj_pos':
            node_attr = torch.hstack([x1, x2])

        return node_attr
    
    def forward(self, data, eval=False, use_edge_att=True):
        data = data.cuda()
        traj = data['traj']
        traj = traj.view(traj.shape[0], -1)
        batch = [torch.tensor([i] * (data._slice_dict['pc_list'][i+1]-data._slice_dict['pc_list'][i]).item()) for i in range(0, data._slice_dict['pc_list'].shape[0]-1)]
        batch = torch.cat(batch).cuda()
        batch_idx = data._slice_dict['pc_list']
        pc = data['pc_list']

        node_attr = self.initial_node_attributes(traj, pc)

        # get edges using knn graph (for computational feasibility)
        if self.graph == 'knn':
            edge_index = knn_graph(x=node_attr, k=self.k, batch=batch)
        elif self.graph == 'radius':
            edge_index = radius_graph(traj, self.r, batch)

        edge_attr = self.initial_edge_attributes(traj, pc, edge_index)

        # get which edges belong to which batch
        batch_edge = torch.zeros(edge_index.shape[1]).cuda()
        for i in range(1, batch_idx.shape[0]-1):
            mask = (edge_index.T < batch_idx[i+1]) & (edge_index.T >= batch_idx[i])
            mask = mask[:, 0] * i
            batch_edge += mask      

        src, dst = edge_index
        # computes per edge index by computing dot product between node features
        if not use_edge_att:
            score = (node_attr[src] * node_attr[dst]).sum(dim=-1)
        # directly uses edge attirbutes
        else:
            score = torch.linalg.norm(edge_attr, dim=1)
        
        if eval:
            score = score.cpu()
            data = data.cpu()

            edges_filtered = list()
            scores_filtered = list()
            for i, e in enumerate(edge_index.T):
                if score[i] < self.cut_edges:
                    edges_filtered.append(e)
                    scores_filtered.append(score[i])

            edge_index = torch.stack(edges_filtered).T
            score = torch.stack(scores_filtered).T
            src, dst = edge_index

            # original score
            score_orig = torch.zeros((data['pc_list'].shape[0], data['pc_list'].shape[0]))
            score_orig[src, dst] = score

            clusters = defaultdict(list)
            cluster_assignment = dict()
            id_count = 0
            import copy
            for i, scores in enumerate(score_orig):
                # get edges
                idxs = (scores > 0).nonzero(as_tuple=True)[0]
                other_ids = list()

                # check if node A in any cluster yet
                to_add = [i] if i not in cluster_assignment.keys() else []
                _id = None if i not in cluster_assignment.keys() else cluster_assignment[i]
                
                # iterate over edges and find clusters that need to be merged
                for idx in idxs:
                    if idx in cluster_assignment.keys() and _id is None:
                        _id = cluster_assignment[idx.item()]
                    elif idx.item() in cluster_assignment.keys():
                        if _id != cluster_assignment[idx.item()]:
                            other_ids.append(cluster_assignment[idx.item()])
                    else:
                        to_add.append(idx.item())

                # if no connected node as well as node i is not in cluster yet
                if _id is None:
                    _id = id_count
                    id_count += 1
                # change cluster ids and merge clusters
                for change_id in set(other_ids):
                    for node in clusters[change_id]:
                        cluster_assignment[node] = _id
                    clusters[_id].extend(copy.deepcopy(clusters[change_id]))
                    del clusters[change_id]
                
                # add nodes that where in no cluster yet
                clusters[_id].extend(to_add)
                for node in to_add:
                    cluster_assignment[node] = _id
            
            clusters_new = defaultdict(list)
            for c, node_list in clusters.items():
                if len(node_list) < self.min_samples:
                    clusters_new[-1].extend(node_list)
                else:
                    clusters_new[c] = node_list
            clusters = clusters_new

            self.visualize(torch.arange(pc.shape[0]), edge_index, pc[:, :-1], clusters)

            clusters = np.array([cluster_assignment[k] for k in sorted(cluster_assignment.keys())])

            return score, clusters, edge_index, batch_edge

        return score, edge_index, batch_edge
    
    def tarjan(self, nodes, scores):
        time = 0
        def SCCUtil(u, low, disc, stackMember, st):
            # Initialize discovery time and low value
            disc[u] = time
            low[u] = time
            time += 1
            stackMember[u] = True
            st.append(u)
    
            # Go through all vertices adjacent to this
            for v in (scores > 0).nonzero(as_tuple=True)[0]:
                
                # If v is not visited yet, then recur for it
                if disc[v] == -1 :
                
                    self.SCCUtil(v, low, disc, stackMember, st)
    
                    # Check if the subtree rooted with v has a connection to
                    # one of the ancestors of u
                    # Case 1 (per above discussion on Disc and Low value)
                    low[u] = min(low[u], low[v])
                            
                elif stackMember[v] == True:
    
                    '''Update low value of 'u' only if 'v' is still in stack
                    (i.e. it's a back edge, not cross edge).
                    Case 2 (per above discussion on Disc and Low value) '''
                    low[u] = min(low[u], disc[v])
    
            # head node found, pop the stack and print an SCC
            w = -1 #To store stack extracted vertices
            if low[u] == disc[u]:
                while w != u:
                    w = st.pop()
                    stackMember[w] = False
                      
        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        disc = [-1] * (nodes.shape[0])
        low = [-1] * (nodes.shape[0])
        stackMember = [False] * (nodes.shape[0])
        st =[]

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(nodes.shape[0]):
            if disc[i] == -1:
                SCCUtil(i, low, disc, stackMember, st)

    def make_symmetric(self, score_orig, mode='minimum'):
        for i in range(score_orig.shape[0]):
            for j in range(i, score_orig.shape[0]):
                if score_orig[i, j] == 0 and score_orig[j, i] == 0:
                    continue
                if mode == 'minimum':
                    score_orig[i, j] = score_orig[j, i] = min(score_orig[i, j], score_orig[j, i])
                elif mode == 'maximum':
                    score_orig[i, j] = score_orig[j, i] = max(score_orig[i, j], score_orig[j, i])
        return score_orig
    
    def visualize(self, nodes, edge_indices, pos, clusters):
        import networkx as nx
        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # make graph
        pos = {n.item(): p for n, p in zip(nodes, pos.cpu().numpy())}
        G = nx.Graph()
        G.add_nodes_from(nodes.numpy())
        G.add_edges_from(edge_indices.T.cpu().numpy())

        colors = [(255, 255, 255)] * nodes.shape[0] 
        col_dict = dict()
        for i, (c, node_list) in enumerate(clusters.items()):
            for node in node_list:
                colors[node] = rgb_colors[i]
            col_dict[c] = rgb_colors[i]

        print(col_dict)
        # save graph
        plt.figure(figsize=(50, 50))
        nx.draw_networkx_edges(G, pos, width=3)
        nx.draw_networkx_nodes(G, pos, node_size=2, node_color=colors)
        plt.axis("off")
        plt.savefig('../../../vis_graph.png', bbox_inches='tight')
        quit()

class SimpleGraphLoss(nn.Module):
    def __init__(self, bce_loss=True) -> None:
        super().__init__()
        
        self.bce_loss = nn.BCELoss().cuda() if bce_loss else bce_loss
        self.max_iter = 2000

    def eval(self, logits, data, edge_index):
        loss = 0
        if self.bce_loss:
            y_categories = data.point_categories.unsqueeze(
                0) == data.point_categories.unsqueeze(0).T
            y_categories = y_categories[
                edge_index[0, :], edge_index[1, :]].type(torch.FloatTensor)
            loss += torch.nn.functional.binary_cross_entropy(
                logits.squeeze(), y_categories.squeeze())
        return loss