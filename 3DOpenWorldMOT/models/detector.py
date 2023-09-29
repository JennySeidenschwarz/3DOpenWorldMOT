import os
import numpy as np
from scipy.spatial.transform import Rotation
from pyarrow import feather
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import logging
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import copy
from .tracking_utils import *


logger = logging.getLogger("Model.Tracker")


class Detector3D():
    def __init__(
            self,
            out_path='out',
            split='val', 
            every_x_frame=1, 
            num_interior=10, 
            overlap=5, 
            av2_loader=None, 
            rank=0,
            precomp_dets=False) -> None:
        
        self.detections = dict()
        self.log_id = -1
        self.split = split
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.num_interior = num_interior
        self.av2_loader = av2_loader
        self.precomp_dets = precomp_dets
        self.rank = rank
        self.name = os.path.basename(out_path)
        self.out_path = os.path.join(out_path,  f'rank_{self.rank}')
    
    def new_log_id(self, log_id):
        # save tracks to feather and reset variables
        if self.log_id != -1:
            self.active_tracks = list()
            self.inactive_tracks = list()
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')

        self.log_id = log_id
        # logger.info(f"New log id {log_id}...")
    
    def get_detections(self, points, traj, clusters, timestamps, log_id,
                       gt_instance_ids, gt_instance_cats, last=False):
        # account for padding in from DistributedTestSampler
        if timestamps.cpu()[0, 0].item() in self.detections.keys():
            if last:
                found = self.to_feather()
                if not found:
                    logger.info(f'No detections found in {log_id}')
            return

        # set new log id
        if self.log_id != log_id:
            self.new_log_id(log_id)

        # iterate over clusters that were found and get detections with their 
        # corresponding flows, trajectories and canonical points
        detections = list()
        if type(clusters) == np.ndarray:
            clusters = torch.from_numpy(clusters).to(self.rank)
        
        if str(gt_instance_ids.device) == 'cpu':
            gt_instance_ids = gt_instance_ids.to(self.rank)
            gt_instance_cats = gt_instance_cats.to(self.rank)
        
        if str(points.device) == 'cpu':
            points = points.to(self.rank)
        
        if str(traj.device) == 'cpu':
            traj = traj.to(self.rank)

        for c in torch.unique(clusters):
            num_interior = torch.sum(clusters==c).item()
            gt_id = (torch.bincount(gt_instance_ids[clusters==c]).argmax()).item()
            gt_cat = (torch.bincount(gt_instance_cats[clusters==c]).argmax()).item()

            # filter if cluster too small
            if num_interior < self.num_interior:
                continue
            # filter if 'junk' cluster
            if c == -1:
                continue
            # get points, bounding boxes
            point_cluster = points[clusters==c]

            # generate new detected trajectory
            traj_cluster = traj[clusters==c]
            detections.append(Detection(
                traj_cluster.cpu(),
                point_cluster.cpu(),
                log_id=log_id,
                timestamps=timestamps.cpu(),
                num_interior=num_interior,
                overlap=self.overlap,
                gt_id=gt_id,
                gt_cat=gt_cat))
        
        self.detections[timestamps.cpu()[0, 0].item()] = detections

        if last:
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')
            
        return detections

    def mat_to_quat(self, mat):
        """Convert a 3D rotation matrix to a scalar _first_ quaternion.
        NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
            we use the scalar FIRST convention.
        Args:
            mat: (...,3,3) 3D rotation matrices.
        Returns:
            (...,4) Array of scalar first quaternions.
        """
        # Convert quaternion from scalar first to scalar last.
        quat_xyzw = Rotation.from_matrix(mat).as_quat()
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        return quat_wxyz
    
    def quat_to_mat(self, quat_wxyz):
        """Convert a quaternion to a 3D rotation matrix.

        NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
            we use the scalar FIRST convention.

        Args:
            quat_wxyz: (...,4) array of quaternions in scalar first order.

        Returns:
            (...,3,3) 3D rotation matrix.
        """
        # Convert quaternion from scalar first to scalar last.
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        mat = Rotation.from_quat(quat_xyzw).as_matrix()
        return mat

    def to_feather(self):
        to_feather(self.detections, self.log_id, self.out_path, self.split, self.rank, self.precomp_dets, self.name)
        self.detections = dict()
        # write_path = os.path.join(self.out_path, self.split, self.log_id, 'annotations.feather') # os.path.join(self.out_path, self.split, 'feathers', f'all_{self.rank}.feather')
        # logger.info(f'wrote {write_path}')
        return True

