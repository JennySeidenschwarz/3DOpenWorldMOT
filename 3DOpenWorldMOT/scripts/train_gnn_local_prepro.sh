#!/bin/sh
HYDRA_FULL_ERROR=1 python3 trainer.py \
	models=GNN \
	job_name=GNN_ORACLE_TORCH_GRAPH \
	data=waymo_traj \
	multi_gpu=False \
	out_path=/dvlresearch/jenny/Documents/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT \
	models.hyperparams.edge_attr=min_mean_max_diffpostrajtime \
	models.hyperparams.node_attr=min_mean_max_vel \
	models.hyperparams.graph_construction=mean_dist_over_time\
	models.hyperparams.k=32 \
	models.hyperparams.k_eval=32 \
	wandb=False \
	models.loss_hyperparams.node_loss=True \
	models.hyperparams.use_node_score=True \
	models.hyperparams.clustering=correlation \
	models.hyperparams.my_graph=True \
	data.trajectory_dir=/dvlresearch/jenny/download_resutls/trajectories \
	data.processed_dir=/dvlresearch/jenny/debug_trajectories_waymo/processed/normal/trajectories \
	data.data_dir=/dvlresearch/jenny/Waymo_Converted_GT \
	data.use_all_points=False \
	data.use_all_points_eval=False\
	data.num_points_eval=8000\
	data.do_process=True\
	training.batch_size=32 \
	training.batch_size_val=1 \
	data.debug=False \
	just_eval=True \
	models.hyperparams.oracle_node=True \
	models.hyperparams.oracle_edge=True \
	data.static_thresh=0.0




