#!/bin/sh
HYDRA_FULL_ERROR=1 python3 trainer.py \
	models=GNN \
	job_name=GNN_RealData \
	data=waymo_traj \
	multi_gpu=False \
	out_path=/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT \
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
	data.trajectory_dir=/dvlresearch/jenny/debug_trajectories_waymo/traj_dataset_waymo/gt_all_egocomp_margin0.6_width25 \
	data.processed_dir=/dvlresearch/jenny/debug_trajectories_waymo/processed_remove_static/normal/gt_all_egocomp_margin0.6_width25 \
	data.data_dir=/dvlresearch/jenny/debug_Waymo_Converted_val/Waymo_Converted \
	data.use_all_points=False \
	data.use_all_points_eval=False\
	data.num_points_eval=32000\
	training.batch_size=32 \
	training.batch_size_val=1 \
	data.debug=True	\
	just_eval=True \
	models.hyperparams.oracle_node=True \
	models.hyperparams.oracle_edge=True \
	models.hyperparams.do_visualize=False \
	tracker_options.num_interior=0 \
	tracker_options.do_associate=True \
	tracker_options.precomp_tracks=True \
	tracker_options.a_threshold=2.0 \
  	tracker_options.i_threshold=2.0 \
	models.hyperparams.layer_sizes_edge.l_1=3 \
	models.hyperparams.layer_sizes_edge.l_2=3 \
	models.hyperparams.layer_sizes_node.l_1=3 \
	models.hyperparams.layer_sizes_node.l_2=3






