import torch
import pytorch3d.loss


class ICPRegistration():
    def __init__(self, active_tracks, av2_loader, registration_len_thresh):
        self.active_tracks = active_tracks
        self.av2_loader = av2_loader
        self.registration_len_thresh = registration_len_thresh
    
    def register(self, max_interior_thresh=50):
        detections = dict()
        init_R = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float().cuda()
        init_s = torch.tensor([1]).float().cuda()
        for j, track in enumerate(self.active_tracks):
            track_dets = list()
            # we start from timestep with most points and then go
            # from max -> end -> start -> max
            max_interior_idx = torch.argmax(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior = torch.max(torch.tensor([d.num_interior for d in track.detections])).item()
            start_in_t0 = torch.atleast_2d(track._get_canonical_points(i=max_interior_idx))
            if len(track) > 1 and max_interior > self.registration_len_thresh:
                max_to_end = range(max_interior_idx, len(track)-1)
                end_to_start = range(len(track)-1, 0, -1)
                start_to_max = range(0, max_interior_idx)
                iterators = [max_to_end, end_to_start, start_to_max]
                increment = [+1, -1, +1]

                SimilarityTransforms = [None] * len(track)
                for iterator, increment in zip(iterators, increment):
                    for i in iterator:
                        # number of interior points at next time stamp
                        num_interior = track.detections[i+increment].num_interior

                        # convert pc from ego frame t=i to ego frame t=i+increment
                        t0 = track.detections[i].timestamps[0, 0].item()
                        t1 = track.detections[i+increment].timestamps[0, 0].item()
                        start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)
                        # start_in_t1 = start_in_t0
                        trans0_in_t1 = (start_in_t1.min(axis=0).values + start_in_t1.max(axis=0).values)/2
                        
                        # get canconcial points at t
                        cano0_in_t0 = torch.atleast_2d(track._get_canonical_points(i=i))
                        cano0_in_t1 = track._convert_time(t0, t1, self.av2_loader, cano0_in_t0)
                        # trans0_in_t1 = (cano0_in_t1.min(axis=0).values + cano0_in_t1.max(axis=0).values)/2

                        # get canonical poitns at t+increment
                        cano1_in_t1 = torch.atleast_2d(track._get_canonical_points(i=i+increment))
                        trans1_in_t1 = (cano1_in_t1.min(axis=0).values + cano1_in_t1.max(axis=0).values)/2
                        
                        # init translation by distance between center points, R unit, s=1
                        init_T = (
                            trans1_in_t1 - trans0_in_t1).float().cuda().unsqueeze(0)
                        init = pytorch3d.ops.points_alignment.SimilarityTransform(
                            R=init_R, T=init_T, s=init_s)

                        if num_interior > max_interior_thresh*0.5 and num_interior > 40:
                            # ICP
                            ICPSolution = pytorch3d.ops.points_alignment.iterative_closest_point(
                                start_in_t1.float().cuda().unsqueeze(0),
                                cano1_in_t1.float().cuda().unsqueeze(0),
                                init_transform=init)

                            ### force rotation around z-axis
                            # ICPSolution.RTs.R[:, 0, 2] = 0
                            # ICPSolution.RTs.R[:, 1, 2] = 0
                            # ICPSolution.RTs.R[:, 2, 2] = 1
                            # ICPSolution.RTs.R[:, 2, 0] = 0
                            # ICPSolution.RTs.R[:, 2, 1] = 0
                            # start_registered = pytorch3d.ops.points_alignment._apply_similarity_transform(
                            #     start_in_t1.cuda().float().unsqueeze(0),
                            #     ICPSolution.RTs.R,
                            #     ICPSolution.RTs.T,
                            #     ICPSolution.RTs.s).squeeze().cpu()

                            ### use registered solution, could be some rotation around z-axis
                            start_registered = torch.atleast_2d(ICPSolution.Xt.cpu().squeeze())

                        else:
                            # if point cloud small just use distance as transform and assume no rotation
                            start_registered = pytorch3d.ops.points_alignment._apply_similarity_transform(
                                    start_in_t1.cuda().float().unsqueeze(0),
                                    init.R,
                                    init.T,
                                    init.s).squeeze()
                            start_registered = torch.atleast_2d(start_in_t0.cpu().squeeze())
                        
                        track.detections[i+increment].update_after_registration(start_registered)

                        # concatenate cano points at t+increment and registered points as
                        # point cloud for next timestamp / final point cloud
                        start_in_t0 = start_registered # torch.cat([start_registered, cano1_in_t1])

            else:
                for i in range(len(track.detections)):
                    points = track._get_canonical_points(i=i)
                    mins, maxs = points.min(axis=0).values, points.max(axis=0).values
                    lwh = maxs - mins
                    translation = (maxs + mins)/2
                    
                    # setting last detection
                    track.detections[i].lwh = lwh
                    track.detections[i].translation = translation
                    track_dets.append(track.detections[i])

            detections[j] = track_dets

        return detections