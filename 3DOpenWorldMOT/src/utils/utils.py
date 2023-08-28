import random
import os
import torch
import numpy as np
import logging
import wandb


def update_params(cfg, params_list, iter):
    cfg.training.optim.base_lr = params_list[iter]['lr']
    cfg.training.optim.weight_decay = params_list[iter]['weight_decay']
    cfg.models.loss_hyperparams.focal_loss_node = params_list[iter]['focal_loss_node']
    cfg.models.loss_hyperparams.focal_loss_edge = params_list[iter]['focal_loss_edge']
    cfg.models.loss_hyperparams.gamma_node = params_list[iter]['gamma_node']
    cfg.models.loss_hyperparams.gamma_edge = params_list[iter]['gamma_edge']
    cfg.models.loss_hyperparams.alpha_node = params_list[iter]['alpha_node']
    cfg.models.loss_hyperparams.alpha_edge = params_list[iter]['alpha_edge']
    cfg.models.loss_hyperparams.node_loss = False #params_list[iter]['node_loss']
    # cfg.models.hyperparams.layer_sizes_edge = params_list[iter]['layer_sizes_edge']
    # cfg.models.hyperparams.layer_sizes_node = params_list[iter]['layer_sizes_node']
    cfg.training.epochs = 15
    cfg.data.percentage_data_train = 0.1 
    cfg.data.percentage_data_val = 0.1

    return cfg, params_list

def sample_params():
    params_list = list()
    for _  in range(30):
        focal_loss = random.choice([True, False])
        params = {
            'lr': 10 ** random.choice([-4, -3.5, -3, -2.5, -2, -1.5, -1]),
            'weight_decay': 10 ** random.choice([-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5]),
            'focal_loss_node': focal_loss,
            'focal_loss_edge': focal_loss,
            'alpha_node': random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'alpha_edge': random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'gamma_node': random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4]),
            'gamma_edge': random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4]),
            'node_loss': random.choice([True, False]),
        }
        dims = sample_dims()
        params['layer_sizes_edge'] = dims
        params['layer_sizes_node'] = dims
        params_list.append(params)

    return params_list

def sample_dims():
    dims = [16, 32, 64, 128]                                                                                                                    
    num_layers =  random.choice([1, 2, 3, 4])                                                                                      
    sampled_dims = [random.choice([16, 32, 64])]                                                                                   
    for i in range(1, num_layers):
        if num_layers > 2 and i == 1:
            dim = random.choice(dims[dims.index(sampled_dims[0]):])                                                      
            sampled_dims.append(dim)                                                                                       
        elif num_layers > 3 and i == 2:
            dim = random.choice(dims[dims.index(sampled_dims[1]):])
            sampled_dims.append(dim)
        else:
            dim = random.choice(dims)
            sampled_dims.append(dim)
    dim_dict = dict()
    for i, d in enumerate(sampled_dims):
        dim_dict[f'l_{i}'] = d
            
    return dim_dict


def initialize(cfg):
    '''HYPER PARAMETER'''
    #print(cfg.training.gpu)
    #os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training.gpu
    torch.manual_seed(1)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    random.seed(1)

    '''CREATE DIR'''
    out_path = os.path.join(cfg.out_path, 'out/')
    os.makedirs(out_path, exist_ok=True)
    experiment_dir = os.path.join(out_path, f'detections_{cfg.data.evaluation_split}/')
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoints_dir = os.path.join(out_path, 'checkpoints/')
    os.makedirs(checkpoints_dir, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(cfg)

    name = get_experiment_name(cfg, logger, checkpoints_dir)

    if cfg.wandb:
        wandb.login(key='3b716e6ab76d92ef92724aa37089b074ef19e29c')
        wandb.init(config=cfg, project=cfg.category, name=name)

    is_neural_net = cfg.models.model_name != 'DBSCAN' \
                and cfg.models.model_name != 'SpectralClustering'\
                    and cfg.models.model_name != 'SimpleGraph'\
                    and cfg.models.model_name != 'DBSCAN_Intersection'

    os.makedirs(experiment_dir + name, exist_ok=True)
    logger.info(f'Detections are stored under {experiment_dir + name}...')
    os.makedirs(str(checkpoints_dir) + name, exist_ok=True)
    logger.info(f'Checkpoints are stored under {str(checkpoints_dir) + name}...')

    return logger, experiment_dir, checkpoints_dir, out_path, name, is_neural_net


def get_experiment_name(cfg, logger, checkpoints_dir):
    if cfg.models.model_name != 'SimpleGraph':
        node = '_NS' if cfg.models.hyperparams.use_node_score else ''
        # cluster = '_' + cfg.models.hyperparams.clustering
        my_graph = f"_MG_{cfg.graph_construction.k}_{cfg.graph_construction.r}" if cfg.graph_construction.my_graph else f'_TG_{cfg.graph_construction.k}_{cfg.graph_construction.r}'
        layer_norm = "_LN_" if cfg.models.hyperparams.layer_norm else ""
        batch_norm = "_BN_" if cfg.models.hyperparams.batch_norm else ""
        drop = "_DR_" if cfg.models.hyperparams.drop_out else ""
        augment = "_AU_" if cfg.models.hyperparams.augment else ""

        # name = cfg.models.hyperparams.graph_construction + '_' + cfg.models.hyperparams.edge_attr + "_" + cfg.models.hyperparams.node_attr + node + my_graph # + cluster
        name = node + my_graph + layer_norm + batch_norm + drop + augment # + cluster
        name = f'{cfg.data.num_points_eval}' + "_" + name if not cfg.data.use_all_points_eval else name
        name = f'{cfg.data.num_points}' + "_" + name if not cfg.data.use_all_points else name
        name = f'{cfg.training.optim.base_lr}' + "_" + name
        name = f'{cfg.training.optim.weight_decay}' + "_" + name
        if cfg.models.loss_hyperparams.focal_loss_node:
            name = f'{cfg.models.loss_hyperparams.gamma_node}' + "_" + name
            name = f'{cfg.models.loss_hyperparams.alpha_node}' + "_" + name
        if cfg.models.loss_hyperparams.focal_loss_edge:
            name = f'{cfg.models.loss_hyperparams.gamma_edge}' + "_" + name
            name = f'{cfg.models.loss_hyperparams.alpha_edge}' + "_" + name
        edge_size = '_'.join([str(v) for v in cfg.models.hyperparams.layer_sizes_edge.values()])
        name = f'{edge_size}' + "_" + name
        node_size = '_'.join([str(v) for v in cfg.models.hyperparams.layer_sizes_node.values()])
        name = f'{node_size}' + "_" + name
        
        name = 'nooracle' + "_" + name if not cfg.models.hyperparams.oracle_node and not cfg.models.hyperparams.oracle_edge else name
        name = 'oracleedge' + "_" + name if cfg.models.hyperparams.oracle_edge else name
        name = 'oraclenode' + "_" + name if cfg.models.hyperparams.oracle_node else name
        name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
        logger.info(f'Using this name: {name}')
        os.makedirs(checkpoints_dir + name, exist_ok=True)
    elif cfg.models.model_name == 'DBSCAN':
        name = cfg.models.hyperparams.input + "_" + str(cfg.models.hyperparams.thresh) + "_" + str(cfg.models.hyperparams.min_samples)
        name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
    elif cfg.models.model_name == 'DBSCAN_Intersection':
        name = cfg.models.hyperparams.input_traj + "_" + str(cfg.models.hyperparams.thresh_traj) + "_" + str(cfg.models.hyperparams.min_samples_traj) + "_" + str(cfg.models.hyperparams.thresh_pos) + "_" + str(cfg.models.hyperparams.min_samples_pos) + "_" + str(cfg.models.hyperparams.flow_thresh)
        name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
    
    name = cfg.job_name + '_' + str(cfg.data.percentage_data_train) + '_' + str(cfg.data.percentage_data_val) + '_' + name

    return name


def close():
    wandb.finish()
    logging.shutdown()