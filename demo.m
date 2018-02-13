
%Load training Data
weight_feature= [15 40 120 10]; 
[collecton_video, Label] = b1_clothes_classification(weight_feature, 13, 42);

%Server for ROS interaction
dataserver = rossvcserver('/matlabnode/sequence_data', 'uchile_srvs/ID', @c1_online_classification)
