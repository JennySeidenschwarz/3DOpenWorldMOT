from .DBSCAN import DBSCAN
from .DBSCAN_Intersection import DBSCAN_Intersection
from .SpectralClustering import SpectralClustering
from .detector import Detector3D
from .GNN import ClusterGNN, GNNLoss
from .SimpleGraph import SimpleGraph, SimpleGraphLoss
from .SimpleTracker import SimpleTracker
from .OracleTracker import OnlineOracleTracker, OfflineOracleTracker
from .icp_registration import ICPRegistration
from .constrained_icp import ConstrainedICPRegistration
from .flow_registration import FlowRegistration
from .collapse import Collapser

_model_factory = {
    'DBSCAN': DBSCAN,
    'DBSCAN_Intersection': DBSCAN_Intersection,
    'GNN': ClusterGNN,
    'SpectralClustering': SpectralClustering,
    'SimpleGraph': SimpleGraph}
_loss_factory = {
    'DBSCAN': None,
    'DBSCAN_Intersection': None,
    'GNN': GNNLoss,
    'SpectralClustering': None,
    'SimpleGraph': SimpleGraphLoss}
_tracker_factory = {
    'SimpleTracker': SimpleTracker,
    'OnlineOracleTracker': OnlineOracleTracker,
    'OfflineOracleTracker': OfflineOracleTracker}
_registration_factory = {
    'ICP': ICPRegistration,
    'Flow': FlowRegistration,
    'ConstICP': ConstrainedICPRegistration
}
_collaps_factory = {
    'SimpleCollaps': Collapser,
}