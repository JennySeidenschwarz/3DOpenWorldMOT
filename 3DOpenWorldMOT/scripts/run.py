import src
from src.utils.utils import sample_params, initialize, close, update_params
import os
import hydra
from omegaconf import OmegaConf
import shutil
from data_utils.TrajectoryDataset import get_TrajectoryDataLoader
import torch
import torch.multiprocessing as mp
from src.training.training import train

@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    iters = 1
    if cfg.training.hypersearch:
        params_list = sample_params()
        iters = 30

    for iter in range(iters):
        if cfg.training.hypersearch:
            cfg = update_params(cfg, params_list, iter)
            print(f"Current params: {params_list[iter]}")

        OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
        logger, experiment_dir, checkpoints_dir, out_path, name, is_neural_net = initialize(cfg)
        
        if os.path.isdir(os.path.join(out_path, 'val', 'feathers')):
            shutil.rmtree(os.path.join(out_path, 'val', 'feathers'))
        
        # needed for preprocessing
        if cfg.data.do_process:
            logger.info("Start processing training data ...")
            _, _, _ = get_TrajectoryDataLoader(cfg)
        
        #  start training
        if cfg.multi_gpu:
            world_size = torch.cuda.device_count()
            in_args = (cfg, world_size, name, logger, experiment_dir, checkpoints_dir, is_neural_net)
            mp.spawn(train, args=in_args, nprocs=world_size, join=True)
        elif torch.cuda.is_available():
            train(0, cfg, 1, name, logger, experiment_dir, checkpoints_dir, is_neural_net)
        else:
            train('cpu', cfg, 1, name, logger, experiment_dir, checkpoints_dir, is_neural_net)

        close()


if __name__ == '__main__':
    main()
