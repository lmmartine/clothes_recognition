
%Load training Data
% weight_feature= [15 40 120 10]; 
weight_feature= [40 40 60 10]; 
[collecton_video, Label] = b1_clothes_classification(weight_feature, 13, 18);

%Server for ROS interaction
dataserver = rossvcserver('/matlab_node/sequence_data', 'uchile_srvs/ID', @c1_online_classification)

global weight_feature
global collecton_video