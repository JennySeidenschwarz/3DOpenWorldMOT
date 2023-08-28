#!/bin/bash
directory_name = "trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25"
directory_name = "trajectories_removestatic1_numcd44_removegroundrc0_remove_ground_pts_patch1_removefar1_removeheight1_len25"

HYDRA_FULL_ERROR=1 python3 scripts/run.py \
	models=GNN \
	job_name=GNN_RealData_ValTrainOLD \
	data=waymo_traj \
	multi_gpu=False \
	out_path=/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT \
	data.trajectory_dir=/dvlresearch/jenny/debug_trajectories_waymo/traj_dataset_waymo/trajectories/trajectories \
	data.processed_dir=/dvlresearch/jenny/debug_trajectories_waymo/processed/normal/trajectories/trajectories \
	data.data_dir=/dvlresearch/jenny/debug_trajectories_waymo/point_clouds_preprocessed \
	models.hyperparams.edge_attr=min_mean_max_diffpostrajtime \
	models.hyperparams.node_attr=min_mean_max_vel \
	graph_construction.graph_construction=mean_dist_over_time \
	graph_construction.k=32 \
	graph_construction.r=2.0 \
	wandb=False \
	models.loss_hyperparams.node_loss=True \
	models.loss_hyperparams.bce_loss=True \
	models.loss_hyperparams.node_weight=1 \
	models.loss_hyperparams.edge_weight=1 \
	models.hyperparams.use_node_score=0.5 \
	models.hyperparams.filter_edges=0.5 \
	models.hyperparams.clustering=correlation \
	graph_construction.my_graph=True \
	models.hyperparams.graph=radius \
	data.use_all_points=False \
	data.use_all_points_eval=False \
	data.num_points_eval=16000\
	data.num_points=16000\
	training.batch_size=4 \
	training.batch_size_val=4 \
	data.debug=True \
	just_eval=False \
	detection_options.precomp_dets=False \
	models.hyperparams.oracle_node=False \
	models.hyperparams.oracle_edge=False \
	models.hyperparams.do_visualize=False \
	models.hyperparams.layer_norm=True \
	models.hyperparams.augment=False \
	detection_options.num_interior=20 \
	training.eval_every_x=1 \
	training.num_workers=0 \
	half_precision=True \
	lr_scheduler.params.step_size=100 \
	lr_scheduler.params.gamma=0.7 \
	training.optim.optimizer.o_class=Adam \
	data.do_process=True \
	data.static_thresh=0.0 \
	models.loss_hyperparams.focal_loss_node=False \
	models.loss_hyperparams.alpha_node=0.9221111 \
	models.loss_hyperparams.gamma_node=3.155 \
	models.loss_hyperparams.focal_loss_edge=False \
	models.loss_hyperparams.alpha_edge=0.929 \
	models.loss_hyperparams.gamma_edge=1.902 \
	training.optim.base_lr=0.06176295901709523 \
	training.optim.weight_decay=3.5900203994472646e-07 \
	models.hyperparams.layer_sizes_edge.l_1=64 \
	models.hyperparams.layer_sizes_node.l_1=64 \
	models.loss_hyperparams.ignore_stat_edges=0 \
	models.loss_hyperparams.ignore_stat_nodes=0 \
	data.filtered_file_path=/dvlresearch/jenny/Waymo_Converted_filtered \



HYDRA_FULL_ERROR=1 python3 scripts/run.py models=GNN job_name=GNN_RealData_ValTrainOLD data=waymo_traj multi_gpu=False out_path=/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT data.trajectory_dir=/dvlresearch/jenny/debug_trajectories_waymo/traj_dataset_waymo/trajectories/trajectories data.processed_dir=/dvlresearch/jenny/debug_trajectories_waymo/processed/normal/trajectories/trajectories data.data_dir=/dvlresearch/jenny/debug_trajectories_waymo/point_clouds_preprocessed models.hyperparams.edge_attr=min_mean_max_diffpostrajtime models.hyperparams.node_attr=min_mean_max_vel graph_construction.graph_construction=mean_dist_over_time graph_construction.k=32 graph_construction.r=2.0 wandb=False models.loss_hyperparams.node_loss=True models.loss_hyperparams.bce_loss=True models.loss_hyperparams.node_weight=1 models.loss_hyperparams.edge_weight=1 models.hyperparams.use_node_score=0.5 models.hyperparams.filter_edges=0.5 models.hyperparams.clustering=correlation graph_construction.my_graph=True models.hyperparams.graph=radius data.use_all_points=False data.use_all_points_eval=False data.num_points_eval=16000 data.num_points=16000 training.batch_size=4 training.batch_size_val=4 data.debug=True just_eval=False detection_options.precomp_dets=False models.hyperparams.oracle_node=False models.hyperparams.oracle_edge=False models.hyperparams.do_visualize=False models.hyperparams.layer_norm=True models.hyperparams.augment=False detection_options.num_interior=20 training.eval_every_x=1 training.num_workers=0 half_precision=True lr_scheduler.params.step_size=100 lr_scheduler.params.gamma=0.7 training.optim.optimizer.o_class=Adam data.do_process=True data.static_thresh=0.0 models.loss_hyperparams.focal_loss_node=False models.loss_hyperparams.alpha_node=0.9221111 models.loss_hyperparams.gamma_node=3.155 models.loss_hyperparams.focal_loss_edge=False models.loss_hyperparams.alpha_edge=0.929 	models.loss_hyperparams.gamma_edge=1.902 	training.optim.base_lr=0.06176295901709523 	training.optim.weight_decay=3.5900203994472646e-07 models.hyperparams.layer_sizes_edge.l_1=64 	models.hyperparams.layer_sizes_node.l_1=64	models.loss_hyperparams.ignore_stat_edges=0 models.loss_hyperparams.ignore_stat_nodes=0 data.filtered_file_path=/dvlresearch/jenny/Waymo_Converted_filtered 

python3 run.py models=GNN job_name=GNN_RealData_ValTrainOLD data=waymo_traj multi_gpu=False out_path=/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT data.trajectory_dir=/dvlresearch/jenny/debug_trajectories_waymo/traj_dataset_waymo/trajectories_removestatic1_numcd4_removegroundrc1_removefar1_removeheight1_len25/trajectories_removestatic1_numcd4_removegroundrc1_removefar1_removeheight1_len25 data.processed_dir=/dvlresearch/jenny/debug_trajectories_waymo/processed/normal/trajectories_removestatic1_numcd4_removegroundrc1_removefar1_removeheight1_len25/trajectories_removestatic1_numcd4_removegroundrc1_removefar1_removeheight1_len25 data.data_dir=/dvlresearch/jenny/debug_trajectories_waymo/point_clouds_preprocessed models.hyperparams.edge_attr=min_mean_max_diffpostrajtime models.hyperparams.node_attr=min_mean_max_vel graph_construction.graph_construction=mean_dist_over_time graph_construction.k=32 graph_construction.r=2.0 wandb=False models.loss_hyperparams.node_loss=True models.loss_hyperparams.bce_loss=True models.loss_hyperparams.node_weight=1 models.loss_hyperparams.edge_weight=1 models.hyperparams.use_node_score=0.5 models.hyperparams.filter_edges=0.5 models.hyperparams.clustering=correlation graph_construction.my_graph=True models.hyperparams.graph=radius data.use_all_points=False data.use_all_points_eval=False data.num_points_eval=16000 data.num_points=16000 training.batch_size=4 training.batch_size_val=4 data.debug=True just_eval=False detection_options.precomp_dets=False models.hyperparams.oracle_node=False models.hyperparams.oracle_edge=False models.hyperparams.do_visualize=False models.hyperparams.layer_norm=True models.hyperparams.augment=False detection_options.num_interior=20 training.eval_every_x=1 training.num_workers=0 half_precision=True lr_scheduler.params.step_size=100 lr_scheduler.params.gamma=0.7 training.optim.optimizer.o_class=Adam data.do_process=True data.static_thresh=0.0 models.loss_hyperparams.focal_loss_node=False models.loss_hyperparams.alpha_node=0.9221111 models.loss_hyperparams.gamma_node=3.155 models.loss_hyperparams.focal_loss_edge=False models.loss_hyperparams.alpha_edge=0.929 models.loss_hyperparams.gamma_edge=1.902 training.optim.base_lr=0.06176295901709523 training.optim.weight_decay=3.5900203994472646e-07 models.hyperparams.layer_sizes_edge.l_1=64 models.hyperparams.layer_sizes_node.l_1=64 models.loss_hyperparams.ignore_stat_edges=0 models.loss_hyperparams.ignore_stat_nodes=0 data.filtered_file_path=/dvlresearch/jenny/Waymo_Converted_filtered 