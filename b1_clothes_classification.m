function [collecton_video, Label] = b1_clothes_classification(weight_feature, imginit, imgend)

% function [collecton_video, Label] = b1_clothes_classification(weight_feature)
% weight_feature

warning off
%pctRunOnAll warning('off','all')
%clear all
%close all
% clc

flag = true;

addpath('./BSplineFitting');
addpath('./Functions');
addpath('./Classification');
addpath('./libSVM');
addpath('./SurfaceFeature');
addpath('./SpatialPyramid');
addpath(genpath([pwd,'/GPML']));
addpath('./ShapeContent');
addpath('./Utilities');
addpath('./vlfeat/toolbox');
addpath(genpath('./RandomForest'));
addpath('./FINDDD');
addpath('./myGP');
addpath('./gentleboost');
addpath('./adaboost');

vl_setup
startup

%% script setting
coding_opt = 'LLC';
para.isnorm = 1;

para.local.bsp = 1;
para.local.finddd =0;
para.local.lbp = 0;
para.local.sc = 0;
para.local.dlcm = 0;
para.local.sift = 0;

para.global.si = 1;
para.global.lbp = 1;
para.global.topo = 1;
para.global.dlcm = 0;
para.global.imm = 0;
para.global.vol = 0;

para.distintic.keyparthist = 0;
para.distintic.keyparthist_onlyneck = 1;
para.distintic.keyparthist_onlywaist = 1;
para.distintic.neckshirt = 0;
para.distintic.size = 0;

current_dir='~/bags';
data_dir = '~/bags/data/';


% category = {'pant','shirt'};
category = {'pant','shirt','tshirt','sweater','towel'};
size_class=3;
size_move=10;

%% main loop
Instance = [];
Label = [];
ClothesID = [];
collecton_video = [];
id = 1;
id_colecction = 1;
for iter_i = 1:length(category)
for iter_j = 1:size_class
for iter_k = 1:size_move

name_local_file = [current_dir,'/Features_select/local_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)];
name_global_file = [current_dir,'/Features_select/global_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)];
name_distintic_file = [current_dir,'/Features_select/distintic_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)];
if exist([name_local_file '.mat'],'file') && exist([name_global_file '.mat'],'file') 
load([name_local_file '.mat']);
load([name_global_file '.mat']);
if exist([name_distintic_file '.mat'],'file')
    load([name_distintic_file '.mat']);
end
    instance = [];
    local_feature = [];
    global_feature = [];
    distintic_feature = [];
    video_instance = [];




    w_bsp1 = weight_feature(1); %40;		
	w_lbp1 = weight_feature(2)/60; %40/60;
	w_si1 = weight_feature(3); %60;
	w_topo1 = weight_feature(4); %10;
	% w_size1 = weight_feature(5)/(640*480); %10;
    w_bsp2 = w_bsp1;		
	w_lbp2 = w_lbp1;
	w_si2 = w_lbp1;
	w_topo2 = w_topo1;
	% w_size2 = w_size1;

    % for iter_l = 1:2
    iter_l =1;
        global_feature_tmp = [];
        local_feature_tmp = [];
        distintic_feature_tmp = [];

        if para.local.bsp
        	vbsp = allfeatures_local(iter_l).dscr_bsp ;
            local_feature_tmp = [ local_feature_tmp, vbsp*w_bsp1];
        end
        if para.global.lbp
        	vlbp=allfeatures_global(iter_l).lbp;
            global_feature_tmp = [ global_feature_tmp, vlbp*w_lbp1];
        end
        if para.global.si
        	vsi=allfeatures_global(iter_l).si;
            global_feature_tmp = [ global_feature_tmp, vsi*w_si1];
        end
        if para.global.topo
        	vtopo =allfeatures_global(iter_l).topo;
            global_feature_tmp = [ global_feature_tmp, vtopo*w_topo1];
        end
        if para.distintic.size
            distintic_feature_tmp = [ distintic_feature_tmp, allfeatures_distintic.size2d(iter_l)*w_size1 ]; % /(640*480)
        end

        local_feature = [ local_feature, local_feature_tmp ];
        global_feature = [ global_feature, global_feature_tmp ];
        distintic_feature = [ distintic_feature, distintic_feature_tmp ];

        frame_instance = [local_feature_tmp, global_feature_tmp, distintic_feature_tmp ];
        video_instance = [video_instance; frame_instance];
    % end

    % for iter_l = 4+imginit:imgend
    % length(allfeatures_local)
    % imgend
    % for iter_l = length(allfeatures_local)-imgend+1: length(allfeatures_local)-imginit +1 %25
    for iter_l = 4 + imginit : 4+imgend

    % for iter_l = length(allfeatures_local)-25: length(allfeatures_local) %25
    % for iter_l = max(5,length(allfeatures_local)-20): length(allfeatures_local)
        global_feature_tmp = [];
        local_feature_tmp = [];
        distintic_feature_tmp = [];

	    if para.local.bsp
	    	vbsp = allfeatures_local(iter_l).dscr_bsp ;
	        local_feature_tmp = [ local_feature_tmp, vbsp*w_bsp2];
	    end
	    if para.global.lbp
	    	vlbp=allfeatures_global(iter_l).lbp;
	        global_feature_tmp = [ global_feature_tmp, vlbp*w_lbp2];
	    end
	    if para.global.si
	    	vsi=allfeatures_global(iter_l).si;
	        global_feature_tmp = [ global_feature_tmp, vsi*w_si2];
	    end
	    if para.global.topo
	    	vtopo =allfeatures_global(iter_l).topo;
	        global_feature_tmp = [ global_feature_tmp, vtopo*w_topo2];
	    end
        if para.distintic.size
            distintic_feature_tmp = [ distintic_feature_tmp, allfeatures_distintic.size2d(iter_l)*w_size2];
        end
        local_feature = [ local_feature, local_feature_tmp ];
        global_feature = [ global_feature, global_feature_tmp ];
        distintic_feature = [ distintic_feature, distintic_feature_tmp ];

        frame_instance = [local_feature_tmp, global_feature_tmp, distintic_feature_tmp  ];
        video_instance = [video_instance; frame_instance];
    end
    % length(allfeatures_local) - max(5,length(allfeatures_local)-20)
    % last_id = length(allfeatures_local) 
    % for iter_l = length(allfeatures_local) : (4+20)
    %     global_feature_tmp = [];
    %     local_feature_tmp = [];
    %     distintic_feature_tmp = [];

    %     if para.local.bsp
    %         local_feature_tmp = [ local_feature_tmp, allfeatures_local(last_id).dscr_bsp ];
    %     end
    %     if para.global.lbp
    %         global_feature_tmp = [ global_feature_tmp, allfeatures_global(last_id).lbp];
    %     end
    %     if para.global.si
    %         global_feature_tmp = [ global_feature_tmp, allfeatures_global(last_id).si  ];
    %     end
    %     if para.global.topo
    %         global_feature_tmp = [ global_feature_tmp, allfeatures_global(last_id).topo ];
    %     end
    %     if para.distintic.size
    %         distintic_feature_tmp = [ distintic_feature_tmp, allfeatures_distintic.size2d(last_id) ];
    %     end
    %     local_feature = [ local_feature, local_feature_tmp ];
    %     global_feature = [ global_feature, global_feature_tmp ];
    %     distintic_feature = [ distintic_feature, distintic_feature_tmp ];

    %     frame_instance = [local_feature_tmp, global_feature_tmp, distintic_feature_tmp  ];
    %     video_instance = [video_instance; frame_instance];
    % end
    collecton_video (id_colecction,:,:) = video_instance;
    id_colecction = id_colecction+1;

    instance = [ local_feature, global_feature, distintic_feature ];
    % instance = [ local_feature_simple, global_feature_simple ];

    Instance = [ Instance; instance ];
    Label = [ Label; iter_i ]; 
    % ClothesID = [ ClothesID; iter_i*100 + iter_j*10 + iter_k];
    ClothesID = [ ClothesID; id];
else
    % [name_local_file '.mat or ' name_global_file '.mat doesnt exist'] 
end

end
    id = id+1;
end
end
