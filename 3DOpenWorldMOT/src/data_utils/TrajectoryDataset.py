import glob
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader
import os
import torch
from pathlib import Path
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import numpy as np
from src.data_utils import point_cloud_handling
from src.data_utils import av2_classes
import os.path as osp
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
import logging
import csv
import glob
import re
from multiprocessing.pool import Pool
from functools import partial
import pytorch3d.ops.points_normals as points_normals
from pyarrow import feather
import av2.utils.io as io_utils
from .splits import get_seq_list
# from torch import multiprocessing as mp
from src.models.GNN_utils import initial_node_attributes, get_graph
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.data_utils.DistributedTestSampler import DistributedTestSampler



logger = logging.getLogger("Model.Dataset")


ARGOVERSE_CLASSES = {v: k for k, v in av2_classes._class_dict.items()}
WAYMO_CLASSES = {'TYPE_UNKNOWN': 0, 'TYPE_VECHICLE': 1, 'TYPE_PEDESTRIAN': 2, 'TYPE_SIGN': 3, 'TYPE_CYCLIST': 4}


class TrajectoryDataset(PyGDataset):
    def __init__(
            self,
            data_dir,
            split,
            trajectory_dir,
            use_all_points,
            num_points,
            remove_static,
            static_thresh,
            debug,
            every_x_frame=1,
            margin=0.6,
            _processed_dir=False,
            do_process=True,
            percentage_data=1,
            evaluation_split='train_gnn',
            filtered_file_path=None,
            graph_construction='pos',
            name=None):
        
        if 'gt' in _processed_dir:
            self.trajectory_dir = Path(os.path.join(trajectory_dir, split))
        else:
            self.trajectory_dir = Path(os.path.join(trajectory_dir))
        self.data_dir = data_dir
        self.remove_static = remove_static
        self.static_thresh = static_thresh
        self._trajectory_dir = trajectory_dir
        self.split = split
        self.use_all_points = use_all_points
        self.num_points = num_points
        if self.split == 'train' or self.split == 'val' or self.split == 'test':
            self.loader = AV2SensorDataLoader(
                data_dir=Path(os.path.join(data_dir, split)),
                labels_dir=Path(os.path.join(data_dir, split)))
        else:
            self.loader = None
        self.every_x_frame = every_x_frame
        self.margin = margin
        self._processed_dir = _processed_dir
        self.do_process =  do_process
        self.percentage_data = percentage_data
        self.filtered_file_path = filtered_file_path
        self.graph_construction = graph_construction
        
        # for debugging
        if debug:
            if split == 'val' and 'Argo' in self.data_dir:
                self.seqs = ['04994d08-156c-3018-9717-ba0e29be8153']
            elif split == 'train' and 'Argo' in self.data_dir:
                self.seqs = ['00a6ffc1-6ce9-3bc3-a060-6006e9893a1a']
            elif split == 'val':
                self.seqs = ['16473613811052081539']
            else:
                self.seqs = ['2400780041057579262']
        else:
            self.seqs = get_seq_list(
                path=os.path.join(data_dir, split),
                mode='train' if 'detector' not in evaluation_split else 'evaluation',
                percentage=self.percentage_data)
        
        if 'detector' in evaluation_split:
            split = 'train' if 'train' in evaluation_split else 'val'
            self.already_evaluated = list()
            # self.already_evaluated = [os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{name}/{split}/*/*')]
        else:
            self.already_evaluated = list()

        self.class_dict = ARGOVERSE_CLASSES if 'Argo' in self.data_dir else WAYMO_CLASSES
        super().__init__()
        
    @property
    def raw_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
            
    @property
    def raw_paths(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [os.path.join(self.trajectory_dir, seq, flow_file)\
            for seq in seqs\
                for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                    if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
    
    @property
    def processed_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file[:-3] + 'pt' for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
            
    @property
    def processed_paths(self):
        if not self.do_process:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.processed_dir) and seq not in self.already_evaluated] # [:16]
            return [os.path.join(self.processed_dir, seq, flow_file)\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.processed_dir, seq))))\
                            if i % self.every_x_frame == 0]
        else:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir) and seq not in self.already_evaluated]
            return [os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt')\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq))))\
                             if i % self.every_x_frame == 0]

    def __len__(self):
        return len(self._processed_paths)
    
    def len(self):
        return self.__len__()

    @property
    def processed_dir(self) -> str:
        if 'gt' in self._processed_dir:
            return Path(os.path.join(self._processed_dir, self.split))
        return Path(os.path.join(self._processed_dir))

    def add_margin(self, label):
        # Add a margin to the cuboids. Some cuboids are very tight 
        # and might lose some points.
        if self.margin:
            label.length_m += self.margin
            label.width_m += self.margin
            label.height_m += self.margin
        return label
    
    def _process(self):
        self._processed_paths = self.processed_paths
        if self.do_process:
            logger.info('Processing...')
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()
            logger.info('Done!')
    
    def process(self, multiprocessing=True):
        # only process what is not there yet
        already_processed = glob.glob(str(self.processed_dir)+'/*/*')
        already_processed = list()
        missing_paths = set(self._processed_paths).difference(already_processed)
        missing_paths = [os.path.join(
            self.trajectory_dir, os.path.basename(os.path.dirname(m)), os.path.basename(m)[:-2] + 'npz')\
                for m in missing_paths]
        
        logger.info(f"Already processed {len(already_processed)},\
                    Missing {len(missing_paths)},\
                    In total {len(self._processed_paths)}")
        self.len_missing = len(missing_paths)

        if self.len_missing and self.loader is None:
            self.loader = AV2SensorDataLoader(
                data_dir=Path(os.path.join(self.data_dir, self.split)),
                labels_dir=Path(os.path.join(self.data_dir, self.split)))

        data_loader = enumerate(missing_paths)            
        if multiprocessing:
            # mp.set_start_method('forkserver')
            # with mp.Pool() as pool:
            #     pool.map(self.process_sweep, data_loader, chunksize=None)
            with Pool() as pool:
                _eval_sequence = partial(self.process_sweep)
                pool.map(_eval_sequence, data_loader)
        else:
            for data in data_loader:
                self.process_sweep(data)
        
    def load_initial_pc(
            self, 
            lidar_fpath: Path,
            index=[0, 1],
            laser_id=[0, 1, 2, 3, 4],
            remove_height=True,
            remove_far=True,
            remove_static=True,
            remove_ground_pts_rc=True,
            remove_non_dirvable=False,
            num_cd_dist_frames=4):
        """Get the lidar sweep from the given sweep_directory.
    ​
        Args:
            sweep_directory: path to middle lidar sweep.
            sweep_index: index of the middle lidar sweep.
            width: +/- lidar scans to grab.
        Returns:
            List of plys with their associated pose if all the sweeps exist.
        """

        # get sweep information
        sweep_df = io_utils.read_feather(lidar_fpath)

        # get pc
        lidar_points_ego = sweep_df[
                list(['x', 'y', 'z'])].to_numpy().astype(np.float64)

        # get mask to filter point cloud
        mask = np.ones(lidar_points_ego.shape[0], dtype=bool)

        if 'Argo' not in self._processed_dir:            
            mask = np.logical_and(mask, sweep_df['laser_id'].isin(laser_id))
            mask = np.logical_and(mask, sweep_df['index'].isin(index))

        if remove_ground_pts_rc:
            mask = np.logical_and(mask, sweep_df['non_ground_pts_rc'])

        # remove non drivable area points (non RoI points)
        if remove_non_dirvable and 'Argo' in self._processed_dir:
            mask = np.logical_and(mask, sweep_df['driveable_area_pts'])
        
        # Remove points above certain height.
        if remove_height:
            mask = np.logical_and(mask, sweep_df['low_pts'] < 4)

        # Remove points beyond certain distance.
        if remove_far:
            mask = np.logical_and(mask, sweep_df['close_pts'] < 80)
        
        if remove_static:
            for i in range(1, num_cd_dist_frames+1):
                mask = np.logical_and(mask, sweep_df[f'cd_dist_{i}'] > 0.2)

        return lidar_points_ego, mask
                
    def process_sweep(self, data):
        j, traj_file = data

        if j % 100 == 0:
            logger.info(f"sweep {j}/{self.len_missing}, {j}-th file")
        
        processed_path = os.path.join(
            self.processed_dir,
            os.path.basename(os.path.dirname(traj_file)),
            os.path.basename(traj_file)[:-3] + 'pt')
        seq = os.path.basename(os.path.dirname(traj_file))
            
        # load original pc
        # orig_path = os.path.join(
        #     self.data_dir,
        #     self.split,
        #     os.path.basename(os.path.dirname(traj_file)),
        #     'sensors',
        #     'lidar',
        #     os.path.basename(traj_file)[:-3] + 'feather')
        # lidar_points_ego, mask = self.load_initial_pc(orig_path)
        # normals = points_normals.estimate_pointcloud_normals(
        #     torch.from_numpy(lidar_points_ego).cuda().unsqueeze(0)).squeeze()
        # pc_normals = normals[mask]

        # load point clouds
        pred = np.load(traj_file)
        traj = pred['traj']
        pc_list = pred['pcs'] if 'pcs' in [k for k in pred.keys()] else pred['pc_list']
        timestamps = pred['timestamps']

        if len(pc_list.shape) > 2:
            pc_list = pc_list[0]

        if len(pred['timestamps'].shape) > 1:
            timestamps = timestamps[0]

        # mean_min_max_vel
        graph_attr = initial_node_attributes(
            torch.from_numpy(traj),
            torch.from_numpy(pc_list),
            'mean_min_max_vel',
            timestamps=torch.from_numpy(timestamps).unsqueeze(0),
            batch=torch.zeros(pc_list.shape[0], dtype=torch.long))

        mean_min_max_vel = get_graph(
            self.graph_construction.k,
            self.graph_construction.r,
            self.graph_construction.graph,
            graph_attr,
            self.graph_construction.my_graph)

        # mean_dist_over_time
        graph_attr = initial_node_attributes(
            torch.from_numpy(traj),
            torch.from_numpy(pc_list),
            self.graph_construction,
            timestamps=torch.from_numpy(timestamps).unsqueeze(0),
            batch=torch.zeros(pc_list.shape[0], dtype=torch.long))

        mean_dist_over_time = get_graph(
            self.graph_construction.k,
            self.graph_construction.r,
            self.graph_construction.graph,
            graph_attr,
            self.graph_construction.my_graph)

        # get labels
        if self.split != 'test':
            labels = self.loader.get_labels_at_lidar_timestamp(
                log_id=seq, lidar_timestamp_ns=int(timestamps[0]))
            if 'Waymo' in self.data_dir:
                filtered_file_path = f'{self.filtered_file_path}/Waymo_Converted_filtered_{self.split}/{self.split}_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
            else:
                filtered_file_path = f'{self.filtered_file_path}/Argoverse2_filtered/{self.split}_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
            labels_mov = self.loader.get_labels_at_lidar_timestamp_all(
                filtered_file_path, log_id=seq, lidar_timestamp_ns=int(timestamps[0]), get_moving=True)
            
            if self.margin and 'Argo' in self.data_dir:
                for label in labels:
                    self.add_margin(label)

            # remove points of labels that are far away (>80,)
            if 'far' in self._trajectory_dir:
                all_centroids = np.asarray([label.dst_SE3_object.translation for label in labels])
                dists_to_center = np.sqrt(np.sum(all_centroids ** 2, 1))
                ind = np.where(dists_to_center <= 80)[0]
                labels = [labels[i] for i in ind]
            
            # hemove points that hare high
            if 'low' in self._trajectory_dir:
                all_centroids = np.asarray([label.dst_SE3_object.translation for label in labels])[:, -1]
                ind = np.where(all_centroids <= 4)[0]
                labels = [labels[i] for i in ind]
            
            # ALL
            # get per point and object masks and bounding boxs and their labels 
            masks = list()
            for label in labels:
                interior = point_cloud_handling.compute_interior_points_mask(
                        pc_list, label.vertices_m)
                int_label = self.class_dict[label.category] if 'Argo' in self.data_dir else int(label.category)
                interior = interior.astype(int) * int_label
                masks.append(interior)

            if len(labels) == 0:
                masks.append(np.zeros(pc_list.shape[0]))
            
            masks = np.asarray(masks).T
        
            # assign unique label and instance to each point
            # label 0 and instance 0 is background
            point_categories = list()
            point_instances = list()
            for j in range(masks.shape[0]):
                if np.where(masks[j]>0)[0].shape[0] != 0:
                    point_categories.append(masks[j, np.where(masks[j]>0)[0][0]])
                    point_instances.append(np.where(masks[j]>0)[0][0]+1)
                else:
                    point_categories.append(0)
                    point_instances.append(0)

            point_instances = np.asarray(point_instances, dtype=np.int64)
            point_categories = np.asarray(point_categories, dtype=np.int64)

            point_categories=torch.atleast_2d(torch.from_numpy(point_categories).squeeze())
            point_instances=torch.atleast_2d(torch.from_numpy(point_instances).squeeze())

            # ONLY MOVING
            # get per point and object masks and bounding boxs and their labels 
            masks = list()
            for label in labels_mov:
                interior = point_cloud_handling.compute_interior_points_mask(
                        pc_list, label.vertices_m)
                int_label = self.class_dict[label.category] if 'Argo' in self.data_dir else int(label.category)
                interior = interior.astype(int) * int_label
                masks.append(interior)

            if len(labels_mov) == 0:
                masks.append(np.zeros(pc_list.shape[0]))
            
            masks = np.asarray(masks).T
        
            # assign unique label and instance to each point
            # label 0 and instance 0 is background
            point_categories_mov = list()
            point_instances_mov = list()
            for j in range(masks.shape[0]):
                if np.where(masks[j]>0)[0].shape[0] != 0:
                    point_categories_mov.append(masks[j, np.where(masks[j]>0)[0][0]])
                    point_instances_mov.append(np.where(masks[j]>0)[0][0]+1)
                else:
                    point_categories_mov.append(0)
                    point_instances_mov.append(0)

            point_instances_mov = np.asarray(point_instances_mov, dtype=np.int64)
            point_categories_mov = np.asarray(point_categories_mov, dtype=np.int64)

            point_categories_mov=torch.atleast_2d(torch.from_numpy(point_categories_mov).squeeze())
            point_instances_mov=torch.atleast_2d(torch.from_numpy(point_instances_mov).squeeze())

            # putting it all together
            data = PyGData(
                pc_list=torch.from_numpy(pc_list),
                traj=torch.from_numpy(traj),
                timestamps=torch.from_numpy(timestamps),
                point_categories_mov=point_categories_mov,
                point_instances_mov=point_instances_mov,
                point_categories=point_categories,
                point_instances=point_instances,
                log_id=seq,
                mean_dist_over_time=mean_dist_over_time,
                mean_min_max_vel=mean_min_max_vel,
                # pc_normals=pc_normals
		)
        else:
            data = PyGData(
                pc_list=torch.from_numpy(pc_list),
                traj=torch.from_numpy(traj),
                timestamps=torch.from_numpy(timestamps),
                log_id=seq,
                mean_dist_over_time=mean_dist_over_time,
                mean_min_max_vel=mean_min_max_vel,
                # pc_normals=pc_normals
		)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        torch.save(data, osp.join(processed_path))
        
    def _remove_static(self, data):
        # remove static points
        mean_traj = data['traj'][:, :, :-1]
        timestamps = data['timestamps']
        # get mean velocity [m/s] along trajectory and check if > thresh
        diff_dist = torch.linalg.norm(
            mean_traj[:, 1, :] - mean_traj[:, 0, :] , axis=1)

        diff_time = timestamps[1] - timestamps[0]
        # bring from nano / mili seconds to seconds
        if 'Argo' in self.data_dir:
            diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
        else:
            diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)

        mean_traj = diff_dist/diff_time
        mean_traj = mean_traj > self.static_thresh

        # if no moving point and not evaluation, sample few random
        if torch.all(~mean_traj):
            idxs = torch.randint(0, mean_traj.shape[0], size=(200, ))
            mean_traj[idxs] = True
        
        # apply mask
        data['pc_list'] = data['pc_list'][mean_traj, :]
        data['traj'] = data['traj'][mean_traj]
        if 'pc_normals' in data.keys:
            data['pc_normals'] = data['pc_normals'][mean_traj]
        
        data['point_instances'] = data['point_instances'].squeeze()[mean_traj]
        data['point_categories'] = data['point_categories'].squeeze()[mean_traj]
        if 'point_categories_mov' in data.keys:
            data['point_instances_mov'] = data['point_instances_mov'].squeeze()[mean_traj]
            data['point_categories_mov'] = data['point_categories_mov'].squeeze()[mean_traj]
        
        return data

    def get(self, idx): 
        path = self._processed_paths[idx]
        data = torch.load(path)

        if self.remove_static and self.static_thresh > 0:
            data = self._remove_static(data)
        else:
            data['point_categories'] = torch.atleast_1d(data['point_categories'].squeeze())
            data['point_instances'] = torch.atleast_1d(data['point_instances'].squeeze())
            if 'point_categories_mov' in data.keys:
                data['point_categories_mov'] = data['point_categories_mov'].squeeze()
                data['point_instances_mov'] = data['point_instances_mov'].squeeze()

        # if you always want same number of points (during training), sample/re-sample
        if not self.use_all_points and data['traj'].shape[0] > self.num_points:
            # initialize mapping to map edges
            mapping = torch.ones(data['pc_list'].shape[0]) * - 1
            
            # sample points
            idxs = torch.randint(0, data['traj'].shape[0], size=(self.num_points, ))
            mask = torch.arange(data['traj'].shape[0])
            mask = torch.isin(mask, idxs)

            # mask data
            data['pc_list'] = data['pc_list'][idxs, :]
            data['traj'] = data['traj'][idxs]
            data['point_categories'] = data['point_categories'][idxs]
            data['point_instances'] = data['point_instances'][idxs]
            if 'point_categories_mov' in data.keys:
                data['point_categories_mov'] = data['point_categories_mov'][idxs]
                data['point_instances_mov'] = data['point_instances_mov'][idxs]
            
            # mask edges by using mapping
            mapping = mapping.int()
            mapping[idx] = torch.arange(data['pc_list'].shape[0]).int()
            for edge_index in ['mean_dist_over_time', 'mean_min_max_vel,']:
                data[edge_index][0, :] = mapping[data[edge_index][0, :]]
                data[edge_index][1, :] = mapping[data[edge_index][1, :]]
                data[edge_index] = data[edge_index][:, data[edge_index][0, :] != -1]
                data[edge_index] = data[edge_index][:, data[edge_index][1, :] != -1]

        data['batch'] = torch.ones(data['pc_list'].shape[0])*idx
        data['timestamps'] = data['timestamps'].unsqueeze(0)
        data['path'] = path

        return data


def get_TrajectoryDataLoader(cfg, name=None, train=True, val=True, test=False, rank=0):
    # get datasets
    if train and not cfg.just_eval:
        train_data = TrajectoryDataset(cfg.data.data_dir + f'_train/' + os.path.basename(cfg.data.data_dir) if 'Argo' not in cfg.data.data_dir else cfg.data.data_dir,
            'train',
            cfg.data.trajectory_dir + '_train',
            cfg.data.use_all_points,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            do_process=cfg.data.do_process,
            _processed_dir=cfg.data.processed_dir + '_train',
            percentage_data=cfg.data.percentage_data_train,
            filtered_file_path=cfg.data.filtered_file_path,
            graph_construction=cfg.graph_construction)
    else:
        train_data = None
    if val:
        if cfg.data.evaluation_split == 'val_gnn' or cfg.data.evaluation_split == 'val_detector':
            split = 'val'
        else:
            split = 'train'
        val_data = TrajectoryDataset(cfg.data.data_dir + f'_{split}/' + os.path.basename(cfg.data.data_dir) if 'Argo' not in cfg.data.data_dir else cfg.data.data_dir,
                split,
                cfg.data.trajectory_dir + f'_{split}',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                every_x_frame=cfg.data.every_x_frame,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir + f'_{split}', 
                percentage_data=cfg.data.percentage_data_val,
                evaluation_split=cfg.data.evaluation_split,
                filtered_file_path=cfg.data.filtered_file_path,
                name=name,
                graph_construction=cfg.graph_construction)
    else:
        val_data = None
    if test:
        split = 'test'
        test_data = TrajectoryDataset(cfg.data.data_dir + f'_{split}/' + os.path.basename(cfg.data.data_dir) if 'Argo' not in cfg.data.data_dir else cfg.data.data_dir,
                split,
                cfg.data.trajectory_dir + f'_{split}',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                every_x_frame=cfg.data.every_x_frame,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir + f'_{split}', 
                percentage_data=1.0,
                filtered_file_path=cfg.data.filtered_file_path,
                graph_construction=cfg.graph_construction)
    else:
        test_data = None

    # get dataloaders 
    if train_data is not None:
        if not cfg.multi_gpu:
            train_loader = PyGDataLoader(
                train_data,
                batch_size=cfg.training.batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=cfg.training.num_workers)
        else:
            train_sampler = DistributedSampler(
                    train_data,
                    num_replicas=torch.cuda.device_count(),
                    drop_last=False,
                    rank=rank, 
                    shuffle=True,
                    seed=5)
            train_loader = PyGDataLoader(
                train_data,
                batch_size=cfg.training.batch_size,
                sampler=train_sampler,
                num_workers=cfg.training.num_workers)
    else:
        train_loader = None

    if val_data is not None:
        if not cfg.multi_gpu:
            val_loader = PyGDataLoader(
                val_data,
                batch_size=cfg.training.batch_size_val,
                num_workers=cfg.training.num_workers)
        else:
            val_sampler = DistributedTestSampler(
                    val_data,
                    num_replicas=torch.cuda.device_count(),
                    rank=rank,
                    shuffle=False,
                    drop_last=False)
            val_loader = PyGDataLoader(
                val_data,
                batch_size=cfg.training.batch_size_val,
                sampler=val_sampler,
                num_workers=cfg.training.num_workers)

    else:
        val_loader = None

    if test_data is not None:
        if not cfg.multi_gpu:
            test_loader = PyGDataLoader(
                test_data,
                batch_size=cfg.training.batch_size_val,
                num_workers=cfg.training.num_workers)
        else:
            test_sampler = DistributedTestSampler(
                    test_data,
                    num_replicas=torch.cuda.device_count(),
                    rank=rank)
            test_loader = PyGDataLoader(
                test_data,
                batch_size=cfg.training.batch_size_val,
                sampler=test_sampler,
                num_workers=cfg.training.num_workers)
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader




