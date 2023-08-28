# CONFIGURATION HANDLING
import os
import torch
from tqdm import tqdm 

import torch.nn as nn
import torch.distributed as dist

from src.models import _model_factory, _loss_factory, Detector3D
from src.data_utils.TrajectoryDataset import get_TrajectoryDataLoader
from TrackEvalOpenWorld.scripts.run_av2_ow import evaluate_av2_ow_MOT
import wandb

# FOR DETECTION EVALUATION
from src.evaluation import eval_detection
from src.evaluation import calc_nmi
from collections import defaultdict
from pyarrow import feather
import shutil
import numpy as np
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def load_model(cfg, logger, rank=0):
    '''MODEL LOADING'''
    model = _model_factory[cfg.models.model_name](rank=rank, **cfg.models.hyperparams)
    if rank == 0:
        logger.info(model)
    criterion = _loss_factory[cfg.models.model_name]
    start_epoch = 0
    optimizer = None

    if 'DBSCAN' not in cfg.models.model_name and cfg.models.model_name != 'SpectralClustering':
        model = model.to(rank)
        criterion = criterion(**cfg.models.loss_hyperparams, rank=rank).to(rank)

        if cfg.models.model_name != 'SimpleGraph':
            try:
                checkpoint = torch.load(cfg.models.weight_path)
                chkpt_new = dict()
                for k, v in checkpoint['model_state_dict'].items():
                    if 'module' in k:
                        chkpt_new[k[7:]] = v
                    else:
                        chkpt_new[k] = v
                checkpoint['model_state_dict'] = chkpt_new
                start_epoch = checkpoint['epoch'] if not cfg.just_eval else start_epoch
                model.load_state_dict(checkpoint['model_state_dict'])
                met = checkpoint['best_metric']
                metric_mode = checkpoint['metric_mode']
                if rank == 0:
                    logger.info(f'Use pretrained model with {metric_mode}: {met}')
            except:
                if cfg.models.weight_path != '':
                    if rank == 0:
                        logger.info(f'Did not find pretrained model with {cfg.models.weight_path}')
                    quit()
                else:
                    if rank == 0:
                        logger.info('No existing model, starting training from scratch...')

            if cfg.training.optim.optimizer.o_class == 'Adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=cfg.training.optim.optimizer.params.lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=cfg.training.optim.weight_decay
                )
            elif cfg.training.optim.optimizer.o_class == 'RAdam':
                optimizer = torch.optim.RAdam(
                    model.parameters(),
                    lr=cfg.training.optim.optimizer.params.lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=cfg.training.optim.weight_decay
                )
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=cfg.training.optim.optimizer.params.lr,
                    momentum=0.9)

    return model, start_epoch, optimizer, criterion


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def train_one_epoch(model, cfg, epoch, logger, optimizer, train_loader,\
                    rank, criterion, scaler, checkpoints_dir, name):
    
    # Adapt learning rate
    lr = max(cfg.training.optim.optimizer.params.lr * (
        cfg.lr_scheduler.params.gamma ** (
            epoch // cfg.lr_scheduler.params.step_size)), cfg.lr_scheduler.params.clip)
    if rank == 0:
        logger.info('Learning rate:%f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Adapt momentum
    '''
    momentum = max(cfg.training.optim.bn_scheduler.params.bn_momentum * (
        cfg.training.optim.bn_scheduler.params.bn_decay ** (
            epoch // cfg.training.optim.bn_scheduler.params.decay_step)), \
                cfg.training.optim.bn_scheduler.params.bn_clip)
    print('BN momentum updated to: %f' % momentum)
    model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
    '''

    # iterate over dataset
    model = model.train()
    if rank == 0:
        logger.info('---- EPOCH %03d TRAINING ----' % (epoch + 1))
    node_loss = torch.zeros(2).to(rank)
    node_acc = torch.zeros(6).to(rank)
    per_class_node_acc = torch.zeros(40).to(rank)
    edge_loss = torch.zeros(2).to(rank)
    edge_acc = torch.zeros(6).to(rank)
    per_class_edge_acc = torch.zeros(40).to(rank)
    num_node_pos = torch.zeros(len(train_loader)).to(rank)
    num_node_neg = torch.zeros(len(train_loader)).to(rank)
    num_edge_pos = torch.zeros(len(train_loader)).to(rank)
    num_edge_neg = torch.zeros(len(train_loader)).to(rank)
    
    collaps_dict = dict()
    for batch, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
    # for batch, data in enumerate(train_loader):
        data = data.to(rank)
        optimizer.zero_grad()
        if cfg.half_precision:
            with torch.cuda.amp.autocast():
                logits, edge_index, batch_edge = model(data)
                loss, log_dict, hist_node, hist_edge = criterion(logits, data, edge_index)
                scaler.scale(loss).backward()
                for p_name, param in model.named_parameters():
                    if torch.isnan(param.grad).any():
                        logger.info('Having nan in gradients....')
                        return None, None, None, None
                scaler.step(optimizer)
                scaler.update()
        else:
            logits, edge_index, batch_edge = model(data)
            loss, log_dict, hist_node, hist_edge = criterion(logits, data, edge_index)
            loss.backward()
            for p_name, param in model.named_parameters():
                try:
                    if torch.isnan(param.grad).any():
                        logger.info(f'Having nan in gradients {p_name}....')
                        logger.info(data)
                        logger.info(data['path'])
                        # for start, end in zip(data._slice_dict['pc_list'][:-1], data._slice_dict['pc_list'][1:]):
                        #     logger.info(data['pc_list'][start:end])
                        #     logger.info(data['traj'][start:end])
                        logger.info(data._slice_dict['pc_list'])
                        return None, None, None, None
                except:
                    print(p_name)
                    quit()
            optimizer.step()

        if torch.isnan(logits[0]).any():
            logger.info(f'Having nan in logits {logits}....')
            return None, None, None, None
        
        if cfg.wandb and not cfg.multi_gpu:
            if hist_node is not None:
                wandb.log({"train histogram node":
                    wandb.Histogram(np_histogram=hist_node), "epoch": epoch})
            if hist_edge is not None:
                wandb.log({"train histogram edge":
                    wandb.Histogram(np_histogram=hist_edge), "epoch": epoch})
        
        if rank == 0 and cfg.wandb:
            for k, v in log_dict.items():
                if 'num' in k:
                    wandb.log({f'{k}': v, "epoch": epoch})
                else:
                    for i in range(int(v.shape[0]/2)):
                        if v[2*i+1]:
                            wandb.log({f'{k} {i}': v[2*i], "epoch": epoch})
        
        if 'train bce loss edge' in log_dict.keys():
            edge_loss += log_dict['train bce loss edge']
        if 'train bce loss node' in log_dict.keys():
            node_loss += log_dict['train bce loss node']
        if 'train accuracy edge' in log_dict.keys():
            edge_acc += log_dict['train accuracy edge']
        if 'train accuracy edges connected to class' in log_dict.keys():
            per_class_edge_acc += log_dict['train accuracy edges connected to class']
        if 'train accuracy node' in log_dict.keys():
            node_acc = log_dict['train accuracy node']
        if 'train accuracy nodes of class' in log_dict.keys():
            per_class_node_acc += log_dict['train accuracy nodes of class']
        if 'train num node pos' in log_dict.keys():
            num_node_pos[batch] = float(log_dict['train num node pos'])
        if 'train num node neg' in log_dict.keys():
            num_node_neg[batch] = float(log_dict['train num node neg'])
        if 'train num edge pos' in log_dict.keys():
            num_edge_pos[batch] = float(log_dict['train num edge pos'])
        if 'train num edge neg' in log_dict.keys():
            num_edge_neg[batch] = float(log_dict['train num edge neg'])
    
    _num_node_pos = torch.tensor([num_node_pos.sum(), num_node_pos.shape[0]]).to(rank)
    _num_node_neg = torch.tensor([num_node_neg.sum(), num_node_neg.shape[0]]).to(rank)
    _num_edge_pos = torch.tensor([num_edge_pos.sum(), num_edge_pos.shape[0]]).to(rank)
    _num_edge_neg = torch.tensor([num_edge_neg.sum(), num_edge_neg.shape[0]]).to(rank)

    if cfg.multi_gpu:
        dist.all_reduce(node_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(edge_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(edge_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_class_edge_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(node_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_class_node_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(_num_node_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(_num_node_neg, op=dist.ReduceOp.SUM)
        dist.all_reduce(_num_edge_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(_num_edge_neg, op=dist.ReduceOp.SUM)
        
    _num_node_neg = torch.round(_num_node_neg[0]/_num_node_neg[1])
    _num_node_pos = torch.round(_num_node_pos[0]/_num_node_pos[1])
    _num_edge_neg = torch.round(_num_edge_neg[0]/_num_edge_neg[1])
    _num_edge_pos = torch.round(_num_edge_pos[0]/_num_edge_pos[1])

    node_loss = round(float(node_loss[0] / node_loss[1]), 2)
    edge_loss = round(float(edge_loss[0] / edge_loss[1]), 2)
    edge_acc = {k: round(float(edge_acc[2*i] / edge_acc[2*i+1]), 2) \
            for i, k in enumerate(['all', 'neg', 'pos']) if edge_acc[2*i+1] != 0}
    per_class_edge_acc = {i: round(float(per_class_edge_acc[2*i] / per_class_edge_acc[2*i+1]), 2) \
            for i in range(int(per_class_edge_acc.shape[0]/2)) if per_class_edge_acc[2*i+1] != 0}
    node_acc = {k: round(float(node_acc[2*i] / node_acc[2*i+1]), 2) \
            for i, k in enumerate(['all', 'neg', 'pos']) if node_acc[2*i+1] != 0}
    per_class_node_acc = {i: round(float(per_class_node_acc[2*i] / per_class_node_acc[2*i+1]), 2) \
            for i in range(int(per_class_node_acc.shape[0]/2)) if per_class_node_acc[2*i+1] != 0}

    if rank == 0 or rank == 'cpu' or not cfg.multi_gpu:
        if 'train bce loss edge' in log_dict.keys():
            logger.info(f'train bce loss edge per epoch: {edge_loss}')
            if cfg.wandb:
                wandb.log({'train bce loss edge per epoch': edge_loss, "epoch": epoch})
        if 'train bce loss node' in log_dict.keys():
            logger.info(f'train bce loss node per epoch: {node_loss}')
            if cfg.wandb:
                wandb.log({'train bce loss node per epoch': node_loss, "epoch": epoch})
        if 'train accuracy edge' in log_dict.keys():
            logger.info(f'train accuracy edge per epoch (all / neg / pos): {set(edge_acc.values())}')
            if cfg.wandb:
                for k, v in edge_acc.items():
                    wandb.log({f'train accuracy {k} edge per epoch': v, "epoch": epoch})
        if 'train accuracy edges connected to class' in log_dict.keys():
            logger.info(f'train accuracy edges per epoch connected to class: {per_class_edge_acc}')
            if cfg.wandb:
                for k, v in per_class_edge_acc.items():
                    wandb.log({f'train accuracy connected to class {k} edge per epoch': v, "epoch": epoch})
        if 'train accuracy node' in log_dict.keys():
            logger.info(f'train accuracy node per epoch (all / neg / pos): {set(node_acc.values())}')
            if cfg.wandb:
                for k, v in node_acc.items():
                    wandb.log({f'train accuracy {k} node per epoch': v, "epoch": epoch})
        if 'train accuracy nodes of class' in log_dict.keys():
            logger.info(f'train accuracy node per epoch per class: {per_class_node_acc}')
            if cfg.wandb:
                for k, v in per_class_node_acc.items():
                    wandb.log({f'train accuracy of class {k} node per epoch': v, "epoch": epoch})
       
        logger.info(f"train num neg / pos nodes on average per batch {_num_node_neg} / {_num_node_pos}")
        logger.info(f"train num neg / pos edges on average per batch {_num_edge_neg} / {_num_edge_pos}")

        if cfg.wandb:
            if not cfg.multi_gpu:
                wandb.log({'train num node pos/neg ratio': wandb.Histogram(
                    np_histogram=np.histogram(num_node_pos.cpu().numpy()/num_node_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})
                wandb.log({'train num edge pos/neg ratio': wandb.Histogram(
                    np_histogram=np.histogram(num_edge_pos.cpu().numpy()/num_edge_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})

        savepath = str(checkpoints_dir) + name + '/latest_model.pth'
        logger.info(f'Saving at {savepath}...')

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
    
    return model, optimizer, criterion, scaler


def eval_one_epoch(model, do_corr_clustering, rank, cfg, val_loader, experiment_dir,\
        name, val_data, is_neural_net, logger, epoch, criterion, optimizer, checkpoints_dir, best_metric):
    
    node_loss = torch.zeros(2).to(rank)
    node_acc = torch.zeros(6).to(rank)
    edge_loss = torch.zeros(2).to(rank)
    edge_acc = torch.zeros(6).to(rank)
    per_class_edge_acc = torch.zeros(40).to(rank)
    per_class_node_acc = torch.zeros(40).to(rank)
    nmis = torch.zeros(2).to(rank)
    num_node_pos = torch.zeros(len(val_loader)).to(rank)
    num_node_neg = torch.zeros(len(val_loader)).to(rank)
    num_edge_pos = torch.zeros(len(val_loader)).to(rank)
    num_edge_neg = torch.zeros(len(val_loader)).to(rank)
    collaps_dict = dict()
    # intialize detector
    if do_corr_clustering:
        detector = Detector3D(
            experiment_dir + name,
            split='train' if 'train' in cfg.data.evaluation_split else 'val',
            every_x_frame=cfg.data.every_x_frame,
            num_interior=cfg.detection_options.num_interior,
            overlap=cfg.detection_options.overlap,
            av2_loader=val_data.loader,
            rank=rank,
            precomp_dets=cfg.detection_options.precomp_dets)

    with torch.no_grad():
        if is_neural_net:
            model = model.eval()
        if rank == 0:
            logger.info('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            logger.info(f'Doing correlation clustering {do_corr_clustering}')
        # Iterate over validation set
        for batch, (data) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
        # for batch, (data) in enumerate(val_loader):

            # compute clusters
            if cfg.half_precision:
                with torch.cuda.amp.autocast():
                    import time
                    logits, all_clusters, edge_index, _ = model(data, eval=True, name=name, corr_clustering=do_corr_clustering)
            else:
                logits, all_clusters, edge_index, _ = model(data, eval=True, name=name, corr_clustering=do_corr_clustering)
            if logits is not None and torch.isnan(logits[0]).any():
                logger.info(f'Having nan in eval logits {logits}....')
                return None, None, None, None

            _nmis = list()
            batch_idx = data._slice_dict['pc_list']
            if do_corr_clustering:
                for g, clusters in enumerate(all_clusters):
                    # continue if we didnt find clusters
                    if not len(clusters):
                        if batch+1 == len(val_loader) and g+1 == len(all_clusters):
                            found = detector.to_feather()
                            if not found:
                                logger.info(f'No detections found in {data.log_id[g]}')
                        continue

                    # compute nmi
                    nmi = calc_nmi.calc_normalized_mutual_information(
                        data['point_instances'].cpu()[batch_idx[g]:batch_idx[g+1]], clusters)
                    _nmis.append(nmi)

                    # generate detections
                    detections = detector.get_detections(
                        data.pc_list[batch_idx[g]:batch_idx[g+1]],
                        data.traj[batch_idx[g]:batch_idx[g+1]],
                        clusters,
                        data.timestamps[g].unsqueeze(0),
                        data.log_id[g],
                        data['point_instances'][batch_idx[g]:batch_idx[g+1]],
                        last= batch+1 == len(val_loader) and g+1 == len(all_clusters))

            if is_neural_net and logits[0] is not None:
                loss, log_dict, hist_node, hist_edge = criterion(logits, data, edge_index, rank, mode='eval')
                if cfg.wandb and not cfg.multi_gpu:
                    if hist_node is not None:
                        wandb.log({"eval histogram node":
                            wandb.Histogram(np_histogram=hist_node), "epoch": epoch})
                    if hist_edge is not None:
                        wandb.log({"eval histogram edge":
                            wandb.Histogram(np_histogram=hist_edge), "epoch": epoch})
            
            if rank == 0 and cfg.wandb:
                for k, v in log_dict.items():
                    if 'num' in k:
                        wandb.log({f'{k}': v, "epoch": epoch})
                    else:    
                        for i in range(int(v.shape[0]/2)):
                            if v[2*i+1]:
                                wandb.log({f'{k} {i}': v[2*i], "epoch": epoch})
            
            if 'eval bce loss edge' in log_dict.keys():
                edge_loss += log_dict['eval bce loss edge']
            if 'eval bce loss node' in log_dict.keys():
                node_loss += log_dict['eval bce loss node']
            if 'eval accuracy edge' in log_dict.keys():
                edge_acc += log_dict['eval accuracy edge']
            if 'eval accuracy edges connected to class':
                per_class_edge_acc += log_dict['eval accuracy edges connected to class']
            if 'eval accuracy node' in log_dict.keys():
                node_acc = log_dict['eval accuracy node']
            if 'eval accuracy nodes of class' in log_dict.keys():
                per_class_node_acc += log_dict['eval accuracy nodes of class']
            if 'eval num node pos' in log_dict.keys():
                num_node_pos[batch] = float(log_dict['eval num node pos'])
            if 'eval num node neg' in log_dict.keys():
                num_node_neg[batch] = float(log_dict['eval num node neg'])
            if 'eval num edge pos' in log_dict.keys():
                num_edge_pos[batch] = float(log_dict['eval num edge pos'])
            if 'eval num edge neg' in log_dict.keys():
                num_edge_neg[batch] = float(log_dict['eval num edge neg'])

            '''
            if rank == 0 and cfg.wandb:
                if 'eval bce loss edge' in log_dict.keys():
                    wandb.log({'eval bce loss edge': log_dict['eval bce loss edge'], "epoch": epoch})
                if 'eval bce loss node' in log_dict.keys():
                    wandb.log({'eval bce loss node': log_dict['eval bce loss node'], "epoch": epoch})
                if 'eval accuracy edge' in log_dict.keys():
                    wandb.log({'eval accuracy edge': log_dict['eval accuracy edge'], "epoch": epoch})
                if 'eval accuracy node' in log_dict.keys():
                    wandb.log({'eval accuracy node': log_dict['eval accuracy node'], "epoch": epoch})
            '''
            if do_corr_clustering:
                nmi = sum(_nmis) / len(_nmis)
                nmis[0] += float(nmi)
                nmis[1] += 1
            
            if is_neural_net and logits[0] is not None:
                '''
                if 'eval bce loss edge' in log_dict.keys():
                    edge_loss[0] += float(
                        log_dict['eval bce loss edge'])
                    edge_loss[1] += 1
                if 'eval bce loss node' in log_dict.keys():
                    node_loss[0] += float(
                        log_dict['eval bce loss node'])
                    node_loss[1] += 1
                if 'eval accuracy edge' in log_dict.keys():
                    edge_acc[0] += float(
                        log_dict['eval accuracy edge'])
                    edge_acc[1] += 1
                if 'eval accuracy node' in log_dict.keys():
                    node_acc[0] += float(
                        log_dict['eval accuracy node'])
                    node_acc[1] += 1
                '''
                if not cfg.multi_gpu:
                    if 'edge num node pos' in log_dict.keys():
                        num_node_pos[batch] += float(
                            log_dict['edge num node pos'])
                    if 'edge num node neg' in log_dict.keys():
                        num_node_neg[batch] += float(
                            log_dict['edge num node neg'])
                    if 'edge num edge pos' in log_dict.keys():
                        num_edge_pos[batch] += float(
                            log_dict['edge num edge pos'])
                    if 'edge num edge neg' in log_dict.keys():
                        num_edge_neg[batch] += float(
                            log_dict['edge num edge neg'])
            _num_node_pos = torch.tensor([num_node_pos.sum(), num_node_pos.shape[0]]).to(rank)
            _num_node_neg = torch.tensor([num_node_neg.sum(), num_node_neg.shape[0]]).to(rank)
            _num_edge_pos = torch.tensor([num_edge_pos.sum(), num_edge_pos.shape[0]]).to(rank)
            _num_edge_neg = torch.tensor([num_edge_neg.sum(), num_edge_neg.shape[0]]).to(rank)
        
        if cfg.multi_gpu:
            if do_corr_clustering:
                dist.all_reduce(nmis, op=dist.ReduceOp.SUM)
            if is_neural_net:
                dist.all_reduce(node_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(edge_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(edge_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(per_class_edge_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(node_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(per_class_node_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(_num_node_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(_num_node_neg, op=dist.ReduceOp.SUM)
                dist.all_reduce(_num_edge_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(_num_edge_neg, op=dist.ReduceOp.SUM)

        if is_neural_net:
            _num_node_neg = torch.round(_num_node_neg[0]/_num_node_neg[1])
            _num_node_pos = torch.round(_num_node_pos[0]/_num_node_pos[1])
            _num_edge_neg = torch.round(_num_edge_neg[0]/_num_edge_neg[1])
            _num_edge_pos = torch.round(_num_edge_pos[0]/_num_edge_pos[1])
            node_loss = round(float(node_loss[0] / node_loss[1]), 2)
            edge_loss = round(float(edge_loss[0] / edge_loss[1]), 2)
            edge_acc = {k: round(float(edge_acc[2*i] / edge_acc[2*i+1]), 2) \
                    for i, k in enumerate(['all', 'neg', 'pos']) if edge_acc[2*i+1] != 0}
            per_class_edge_acc = {i: round(float(per_class_edge_acc[2*i] / per_class_edge_acc[2*i+1]), 2) \
                    for i in range(int(per_class_edge_acc.shape[0]/2)) if per_class_edge_acc[2*i+1] != 0}
            node_acc = {k: round(float(node_acc[2*i] / node_acc[2*i+1]), 2) \
                    for i, k in enumerate(['all', 'neg', 'pos']) if node_acc[2*i+1] != 0}
            per_class_node_acc = {i: round(float(per_class_node_acc[2*i] / per_class_node_acc[2*i+1]), 2) \
                    for i in range(int(per_class_node_acc.shape[0]/2)) if per_class_node_acc[2*i+1] != 0}
        
        if do_corr_clustering:
            nmis = round(float(nmis[0] / nmis[1]), 2)
        
        if rank == 0 or rank == 'cpu' or not cfg.multi_gpu:
            if do_corr_clustering:
                logger.info(f'nmi: {nmis}')
            
            if is_neural_net:
                if 'eval bce loss edge' in log_dict.keys():
                    logger.info(f'eval bce loss edge per epoch: {edge_loss}')
                    if cfg.wandb:
                        wandb.log({'eval bce loss edge per epoch': edge_loss, "epoch": epoch})
                if 'eval bce loss node' in log_dict.keys():
                    logger.info(f'eval bce loss node per epoch: {node_loss}')
                    if cfg.wandb:
                        wandb.log({'eval bce loss node per epoch': node_loss, "epoch": epoch})
                if 'eval accuracy edge' in log_dict.keys():
                    logger.info(f'eval accuracy edge per epoch (all / neg / pos): {set(edge_acc.values())}')
                    if cfg.wandb:
                        for k, v in edge_acc.items():
                            wandb.log({f'eval accuracy {k} edge per epoch': v, "epoch": epoch})
                if 'eval accuracy edges connected to class' in log_dict.keys():
                    logger.info(f'eval accuracy edges per epoch connected to class: {per_class_edge_acc}')
                    if cfg.wandb:
                        for k, v in per_class_edge_acc.items():
                            wandb.log({f'eval accuracy class {k} edge per epoch': v, "epoch": epoch})
                if 'eval accuracy node' in log_dict.keys():
                    logger.info(f'eval accuracy node per epoch (all / neg / pos): {set(node_acc.values())}')
                    if cfg.wandb:
                        for k, v in node_acc.items():
                            wandb.log({f'eval accuracy {k} node per epoch': v, "epoch": epoch})
                if 'eval accuracy nodes of class' in log_dict.keys():
                    logger.info(f'eval accuracy of nodes per epoch per class: {per_class_node_acc}')
                    if cfg.wandb:
                        for k, v in per_class_node_acc.items():
                            wandb.log({f'eval accuracy class {k} node per epoch': v, "epoch": epoch})
                
                logger.info(f"eval num neg / pos nodes on average per batch {_num_node_neg} / {_num_node_pos}")
                logger.info(f"eval num neg / pos edges on average per batch {_num_edge_neg} / {_num_edge_pos}")
            
            if cfg.wandb:
                if not cfg.multi_gpu:
                    wandb.log({'eval num node pos/neg ratio': wandb.Histogram(
                        np_histogram=np.histogram(num_node_pos.cpu().numpy()/num_node_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})
                    wandb.log({'eval num edge pos/neg ratio': wandb.Histogram(
                        np_histogram=np.histogram(num_edge_pos.cpu().numpy()/num_edge_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})

            if do_corr_clustering:
                # get sequence list for evaluation
                detector_dir = os.path.join(detector.out_path, detector.split)
                try:
                    seq_list = os.listdir(detector_dir)
                except:
                    seq_list = list()
            
            if do_corr_clustering:
                # average NMI
                cluster_metric = [nmis]
                logger.info(f'NMI: {cluster_metric[0]}')
                logger.info(f'Evaluating detection performance...')
                # evaluate detection
                _, detection_metric = eval_detection.eval_detection(
                    gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir),
                    trackers_folder=detector_dir,
                    seq_to_eval=seq_list,
                    remove_far=True,#'80' in cfg.data.trajectory_dir,
                    remove_non_drive='non_drive' in cfg.data.trajectory_dir,
                    remove_non_move=cfg.data.remove_static_gt,
                    remove_non_move_strategy=cfg.data.remove_static_strategy,
                    remove_non_move_thresh=cfg.data.remove_static_thresh,
                    classes_to_eval='all',
                    debug=cfg.data.debug,
                    name=name)
            
                # log metrics
                for_logs = {met: m for met, m in zip(['AP', 'ATE', 'ASE', 'AOE' ,'CDS'], detection_metric)}
            
                if cfg.wandb:
                    for met, m in for_logs.items():
                        wandb.log({met: m, "epoch": epoch})
                    wandb.log({'NMI': cluster_metric[0], "epoch": epoch})
            
            if not is_neural_net and cfg.metric == 'acc':
                metric = detection_metric
            elif cfg.metric == 'acc':
                metric = [edge_acc['all']]
            elif cfg.metric == 'cluster':
                metric = cluster_metric
            else:
                metric = detection_metric

            # store weights if neural net                
            if metric[0] >= best_metric:
                best_metric = metric[0]
                if is_neural_net and not cfg.just_eval:
                    savepath = str(checkpoints_dir) + name + '/best_model.pth'
                    logger.info('Saving at %s...' % savepath)
                    if do_corr_clustering:
                        state = {
                            'epoch': epoch,
                            'ACC': edge_acc['all'],
                            'best_metric': best_metric,
                            'metric_mode': cfg.metric,
                            'NMI': cluster_metric[0],
                            'class_avg_iou': detection_metric[0],
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                    else:
                        state = {
                            'epoch': epoch,
                            'ACC': edge_acc,
                            'best_metric': best_metric,
                            'metric_mode': cfg.metric,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                    torch.save(state, savepath)
            
            if do_corr_clustering:
                logger.info(f'Best {cfg.metric} metric: {best_metric}, acc: {edge_acc}, cluster metric: {cluster_metric}, detection metric: {for_logs}')
            else:
                logger.info(f'Best {cfg.metric} metric: {best_metric}, acc: {edge_acc}')
    
    return model, optimizer, criterion


def train(rank, cfg, world_size, name, logger, experiment_dir, checkpoints_dir, is_neural_net):
    # set half precision
    if cfg.half_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # initialize distributed training
    if cfg.multi_gpu and rank != 'cpu':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = f'{world_size}'
        os.environ['RANK'] = f'{rank}'
        torch.cuda.set_device(rank)
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # load model
    model, start_epoch, optimizer, criterion = \
        load_model(cfg, logger, rank)
    
    # get datasets
    cfg.data.do_process = False
    train_loader, val_loader, test_loader = get_TrajectoryDataLoader(cfg, name=experiment_dir + name, rank=rank)

    if train_loader is not None and rank == 0:
        logger.info("The number of training data is: %d" % len(train_loader.dataset))
    if val_loader is not None and rank == 0:
        logger.info("The number of test data is: %d" % len(val_loader.dataset))

    if cfg.multi_gpu and is_neural_net:
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    
    best_metric = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        if cfg.multi_gpu:
            train_loader.train_sampler.set_epoch(epoch)
            val_loader.val_sampler.set_epoch(epoch)

        if not cfg.just_eval:
            '''Train on chopped scenes'''
            if rank == 0:
                logger.info('**** Epoch (%d/%s) ****' % (
                     epoch + 1, cfg.training.epochs))

            if is_neural_net:
                model, optimizer, criterion, scaler = train_one_epoch(
                    model,
                    cfg,
                    epoch,
                    logger,
                    optimizer,
                    train_loader,
                    rank,
                    criterion,
                    scaler,
                    checkpoints_dir,
                    name)
            if model is None:
                logger.info("Terminating training due to nan values...")
                return
        
        # evaluate
        if epoch % cfg.training.eval_every_x == 0:
            # do corr clustering every eval_corr_every_x epochs if epoch not 0
            do_corr_clustering = epoch % cfg.training.eval_corr_every_x == 0 and epoch != 0
            # do corr clustering if only eval
            do_corr_clustering = do_corr_clustering or cfg.just_eval
            # do corr clustering in last epoch always
            # do_corr_clustering = do_corr_clustering or epoch == cfg.training.epochs - 1

            model, optimizer, criterion = eval_one_epoch(
                model,
                do_corr_clustering,
                rank,
                cfg,
                val_loader,
                experiment_dir,
                name,
                val_loader.dataset,
                is_neural_net, 
                logger,
                epoch,
                criterion,
                optimizer,
                checkpoints_dir,
                best_metric)

        if not is_neural_net or cfg.just_eval:
            break
   
    # final_evaluation
    final_evaluation(
            model,
                do_corr_clustering,
                rank,
                cfg,
                val_loader,
                experiment_dir,
                name,
                val_loader.dataset,
                is_neural_net,
                logger,
                epoch,
                criterion,
                optimizer,
                checkpoints_dir,
                best_metric)


def final_evaluation(model,
                do_corr_clustering,
                rank,
                cfg,
                val_loader,
                experiment_dir,
                name,
                val_data,
                is_neural_net,
                logger,
                epoch,
                criterion,
                optimizer,
                checkpoints_dir,
                best_metric):
    
    # FINAL EVALUATION WITH BEST WEIGHTS
    if is_neural_net:
        if rank == 0:
            logger.info('**** FINAL EVALUATION ****')
        best_model_path = str(checkpoints_dir) + name + '/best_model.pth'
        checkpoint = torch.load(best_model_path)
        chkpt_new = dict()
        for k, v in checkpoint['model_state_dict'].items():
            if 'module' in k:
                 chkpt_new[k[7:]] = v
            else:
                chkpt_new[k] = v
        checkpoint['model_state_dict'] = chkpt_new
        start_epoch = checkpoint['epoch'] if not cfg.just_eval else start_epoch
        # model.load_state_dict(checkpoint['model_state_dict'])

    eval_one_epoch(model, True, rank, cfg, val_loader, experiment_dir,\
            name, val_data, is_neural_net, logger, epoch, criterion, optimizer, checkpoints_dir, best_metric)
