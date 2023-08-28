#!/bin/sh
HYDRA_FULL_ERROR=1 python3 trainer.py \
	models=GNN \
	job_name=GNN_ORACLE_TORCH_GRAPH \
	data=waymo_traj \
	multi_gpu=True \
	out_path=/workspace/result \
	models.hyperparams.edge_attr=min_mean_max_diffpostrajtime \
	models.hyperparams.node_attr=min_mean_max_vel \
	models.hyperparams.graph_construction=mean_dist_over_time\
	models.hyperparams.k=32 \
	models.hyperparams.k_eval=32 \
	wandb=True \
	models.loss_hyperparams.node_loss=True \
	models.hyperparams.use_node_score=True \
	models.hyperparams.clustering=correlation \
	models.hyperparams.my_graph=True \
	data.trajectory_dir=/workspace/gt_all_egocomp_margin0.6_width25 \
	data.processed_dir=/workspace/gt_all_egocomp_margin0.6_width25 \
	data.data_dir=/workspace/Waymo_Converted_val/Waymo_Converted \
	data.use_all_points=False \
	data.use_all_points_eval=False\
	data.num_points_eval=8000\
	training.batch_size=16 \
	training.batch_size_val=1 \
	data.debug=False \
	just_eval=False \
	models.hyperparams.oracle_node=False \
	models.hyperparams.oracle_edge=False






