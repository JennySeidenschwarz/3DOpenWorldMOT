import sklearn.cluster
import torch.nn as nn
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from torch import multiprocessing as mp


class DBSCAN_Intersection():
    def __init__(
            self,
            rank=0,
            min_samples_pos=2,
            thresh_pos=6,
            min_samples_traj=2,
            thresh_traj=6,
            input_traj='traj',
            flow_thresh=0.2,
            dataset='waymo',
            min_samples_after=20,
            visualization=False) -> None:

        self.model_pos = sklearn.cluster.DBSCAN(
            min_samples=min_samples_pos, eps=thresh_pos, n_jobs=-1)
        self.model_traj = sklearn.cluster.DBSCAN(
            min_samples=min_samples_traj, eps=thresh_traj, n_jobs=-1)
        # mp.set_start_method('forkserver')
        self.input_traj = input_traj
        self.flow_thresh = flow_thresh
        self.dataset = dataset
        self.min_samples_after = min_samples_after
        self.visualization = visualization
    
    def forward(self, clustering, multiprocessing=False):
        batch_idx = clustering._slice_dict['pc_list']
        self.clustering = clustering
        data_loader = enumerate(zip(batch_idx[:-1], batch_idx[1:]))
        if multiprocessing:
            batch_idx = clustering._slice_dict['pc_list']
            data_loader = enumerate(zip(batch_idx[:-1], batch_idx[1:]))
            self.clustering = clustering
            labels = list()
            with mp.Pool() as pool:
                _labels = pool.map(self.cluster, data_loader, chunksize=None)
                labels.append(_labels)
        else:
            data = 0, (batch_idx[:-1], batch_idx[1:])
            labels = self.cluster(data)
            labels = [labels]

        return None, labels, None, None
    
    def cluster(self, data):
        i, (start, end) = data
        clustering = self.clustering
        traj = clustering.traj[start:end].numpy()
        pc = clustering['pc_list'][start:end].numpy()
        if len(pc.shape) != 2:
            pc = pc[0]
        timestamps = clustering['timestamps'][i].squeeze().numpy()

        # if no moving point
        if traj.shape[0] == 0:
            return []

        if self.input_traj == 'traj' or self.input_traj == 'MMMV':
            diff_traj = traj[:, :-1] - traj[:, 1:]

            # get mask to remove static points
            time = timestamps[1:traj.shape[1]]-timestamps[:traj.shape[1]-1]
            if 'waymo' in self.dataset:
                time = time / np.power(10, 6.0) 
            else:
                time = time / np.power(10, 9.0)

            diff_traj = diff_traj / np.expand_dims(time, axis=1)
            mask = np.linalg.norm(diff_traj[:, :, :-1], ord=2, axis=2)

            time = np.expand_dims(time, axis=1)
            time = np.tile(time, (1, 3))
            if self.input_traj == 'MMMV':
                inp_traj = np.vstack([
                    mask.min(axis=-1),
                    mask.max(axis=-1),
                    mask.mean(axis=-1)]).T
            else:
                inp_traj = diff_traj.reshape(diff_traj.shape[0], -1)
            mask = mask.mean(axis=1)
        
        elif self.input_traj == 'scene_flow':
            traj = traj[:, 1, :]
            # get mask to remove static points
            mask = np.linalg.norm(traj, ord=2, axis=1)
            time = timestamps[1]-timestamps[0]
            if 'waymo' in self.dataset:
                time = time / np.power(10, 6.0) 
            else:
                time = time / np.power(10, 9.0) 
            mask = mask / time
            traj = traj / time
            inp_traj = traj.reshape(traj.shape[0], -1)

        labels = np.ones(mask.shape) * -1
        
        # filter pos und traj
        idxs = np.where(mask > self.flow_thresh)[0]
        mask = mask > self.flow_thresh
        inp_pos = pc.reshape(pc.shape[0], -1)
        inp_pos = inp_pos[mask]
        inp_traj = inp_traj[mask]

        # if no points left
        if inp_traj.shape[0] == 0:
            return labels
        
        # get clustering
        clustering_pos = self.model_pos.fit(inp_pos) # only flow 0.0015
        labels_pos = clustering_pos.labels_
        if self.visualization:
            self.visualize(inp_pos, labels_pos, 'position', timestamps[0])

        clustering_traj = self.model_traj.fit(inp_traj) # only flow 0.0015
        labels_traj = clustering_traj.labels_
        if self.visualization:
            self.visualize(inp_pos, labels_traj, 'traj', timestamps[0])
        '''
        # take intersection
        label_count = defaultdict(int)
        i = 0
        for c_pos in labels_pos.unique():
            if c_pos == -1:
                continue
            pos_mask = labels_pos == c_pos
            for c_traj in labels_traj.unique():
                if c_traj == -1:
                    continue
                traj_mask = labels_traj == c_traj
                intersection = np.logical_and(pos_mask, traj_mask)
                labels[intersection] = i
                label_count[i] = intersection.sum()
                i += 1

        if self.min_samples_after > 1:
            for lab, count in label_count.items():
                if count < self.min_samples_after:
                    labels[labels==lab] = -1

        '''
        # take intersection
        cluster_to_id = dict()
        label_count = defaultdict(int)
        i = 0
        for j, (c_pos, c_traj) in enumerate(zip(labels_pos, labels_traj)):
            if str(c_pos) + '_' + str(c_traj) not in cluster_to_id.keys():
                if c_pos == -1 or c_traj == -1:
                    cluster_to_id[str(c_pos) + '_' + str(c_traj)] = -1
                else:
                    cluster_to_id[str(c_pos) + '_' + str(c_traj)] = i
                    i += 1
            label = cluster_to_id[str(c_pos) + '_' + str(c_traj)]
            labels[idxs[j]] = label
            label_count[label] += 1
        
        for lab, count in label_count.items():
            if count < self.min_samples_after:
                labels[labels==lab] = -1
        

        if self.visualization:
            self.visualize(pc.reshape(pc.shape[0], -1), labels, 'combined', timestamps[0])
        labels = labels.astype(int)
        return labels
    
    def visualize(self, position, clustering, name, time):
        os.makedirs('../../../vis_intersection', exist_ok=True)
        import matplotlib.pyplot as plt

        def get_cmap(n, name='hsv'):
            '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(np.unique(clustering).shape[0])
        for i, cluster in enumerate(np.unique(clustering)):
            if cluster == -1:
                continue
            pos = position[clustering==cluster][:-1]
            plt.scatter(pos[:, 0], pos[:, 1], color=cmap(i), s=0.5)
        plt.axis('off')
        plt.savefig(os.path.join('../../../vis_intersection', f'{str(time)}_{name}.jpg'))
        plt.close()

    def __call__(self, clustering, eval=False, name='General', corr_clustering=False):
        return self.forward(clustering)
