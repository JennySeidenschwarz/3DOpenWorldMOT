#!/bin/bash
directory_name = "trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25"
directory_name = "trajectories_removestatic1_numcd44_removegroundrc0_remove_ground_pts_patch1_removefar1_removeheight1_len25"

HYDRA_FULL_ERROR=1 python3 trainer.py \
	models=GNN \
	job_name=GNN_RealData_HS2 \
	data=waymo_traj \
	multi_gpu=False \
	out_path=/workspace/result \
	data.trajectory_dir=/workspace/all_egocomp_margin0.6_width25 \
	data.processed_dir=/workspace/all_egocomp_margin0.6_width25 \
	data.data_dir=/workspace/Waymo_Converted_val/Waymo_Converted \
	models.hyperparams.edge_attr=min_mean_max_diffpostrajtime \
	models.hyperparams.node_attr=min_mean_max_vel \
	models.hyperparams.graph_construction=mean_dist_over_time \
	models.hyperparams.k=16 \
	models.hyperparams.k_eval=16 \
	wandb=True \
	models.loss_hyperparams.node_loss=True \
	models.loss_hyperparams.bce_loss=True \
	models.loss_hyperparams.node_weight=1 \
	models.loss_hyperparams.edge_weight=1 \
	models.hyperparams.use_node_score=True \
	models.hyperparams.clustering=correlation \
	models.hyperparams.my_graph=True \
	models.hyperparams.graph=radius \
	data.use_all_points=False \
	data.use_all_points_eval=False \
	data.num_points_eval=8000\
	data.num_points=4096\
	training.batch_size=16 \
	training.batch_size_val=16 \
	data.debug=True	\
	just_eval=False \
	models.hyperparams.oracle_node=False \
	models.hyperparams.oracle_edge=True \
	models.hyperparams.do_visualize=False \
	tracker_options.num_interior=20 \
	training.eval_per_seq=5 \
	lr_scheduler.params.step_size=20 \
	lr_scheduler.params.gamma=0.7 \
	training.optim.optimizer.o_class=Adam \
	data.do_process=False \
	data.static_thresh=0.0 \
	models.loss_hyperparams.focal_loss_node=True \
	models.loss_hyperparams.alpha_node=0.75 \
	models.loss_hyperparams.gamma_node=3.5 \
	models.loss_hyperparams.focal_loss_edge=False \
	training.optim.base_lr=0.05 \
	training.optim.weight_decay=0.0000015 \
	models.hyperparams.layer_sizes_edge.l_1=32 \
	models.hyperparams.layer_sizes_edge.l_2=32 \
	models.hyperparams.layer_sizes_node.l_1=32 \
	models.hyperparams.layer_sizes_node.l_2=32
