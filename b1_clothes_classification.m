function [collecton_video, Label] = b1_clothes_classification(weight_feature, weight_feature2, imginit, imgend)

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
para.local.finddd =1;
para.local.lbp = 0;
para.local.sc = 1;
para.local.dlcm = 1;
para.local.sift = 1;

para.global.si = 1;
para.global.lbp = 1;
para.global.topo = 1;
para.global.dlcm = 0;
para.global.imm = 0;
para.global.vol = 0;

% para.distintic.keyparthist = 0;
% para.distintic.keyparthist_onlyneck = 1;
% para.distintic.keyparthist_onlywaist = 1;
% para.distintic.neckshirt = 0;
para.distintic.size = 1;

current_dir='~/bags';
data_dir = '~/bags/data/';


category = {'pant','shirt','tshirt','sweater','towel'};
% category = {'towel'};

size_class=3;
size_move=30;

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

name_local_file = [current_dir,'/Features/local_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)];
name_global_file = [current_dir,'/Features/global_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)];
name_distintic_file = [current_dir,'/Features/distintic_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)];
if exist([name_local_file '.mat'],'file') && exist([name_global_file '.mat'],'file') 
load([name_local_file '.mat']);
load([name_global_file '.mat']);
if exist([name_distintic_file '.mat'],'file')
    load([name_distintic_file '.mat']);
end

% name_local_file
    instance = [];
    local_feature = [];
    global_feature = [];
    distintic_feature = [];
    video_instance = [];

% 40    60    80    60    

    w_bsp1 = weight_feature(1); %40;		
    w_finddd1 = weight_feature(2); %10;
    w_sc1 = weight_feature(3); %10;
    w_dlcm1 = weight_feature(4); %10;
    w_sift1 = weight_feature(5); %10;
    w_topo1 = weight_feature(6); %60;
	w_lbp1 = weight_feature(7)/60; %40/60;
	w_si1 = weight_feature(8); %80;
    w_dlcmb1 = weight_feature(9);
    w_imm1 = weight_feature(10);
	w_size1 = weight_feature(11)/(640*480); %180/(640*480);


    w_bsp2 = weight_feature2(1);
    w_finddd2 = weight_feature2(2);
    w_sc2 = weight_feature2(3);
    w_dlcm2 = weight_feature2(4);
    w_sift2 = weight_feature2(5);		
    w_topo2 = weight_feature2(6);
	w_lbp2 = weight_feature2(7)/60;
	w_si2 = weight_feature2(8)/60;%0.67;
    w_dlcmb2 = weight_feature2(9);
    w_imm2 = weight_feature2(10);
	w_size2 = weight_feature2(11)/(640*480);

    % for iter_l = 2:3
    iter_l =2;
        global_feature_tmp = [];
        local_feature_tmp = [];
        distintic_feature_tmp = [];

        if para.local.bsp
            vbsp = allfeatures_local(iter_l).dscr_bsp ;
            %length(vbsp)
            local_feature_tmp = [ local_feature_tmp, vbsp*w_bsp1];
        end
        if para.local.finddd
            vfinddd = allfeatures_local(iter_l).dscr_finddd ;
            %length(vfinddd)
            local_feature_tmp = [ local_feature_tmp, vfinddd*w_finddd1];
        end
        if para.local.sc
            vsc = allfeatures_local(iter_l).dscr_sc ;
            %length(vsc)
            local_feature_tmp = [ local_feature_tmp, vsc*w_sc1];
        end
        if para.local.dlcm
            vdlcm = allfeatures_local(iter_l).dscr_dlcm ;
            %length(vdlcm)
            local_feature_tmp = [ local_feature_tmp, vdlcm*w_dlcm1];
        end
        if para.local.sift
            vsift = allfeatures_local(iter_l).dscr_sift ;
            %length(vsift)
            local_feature_tmp = [ local_feature_tmp, vsift*w_sift1];
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
        if para.global.dlcm
            vdlcm =allfeatures_global(iter_l).dlcm;
            global_feature_tmp = [ global_feature_tmp, vdlcm*w_dlcmb1];
        end
        if para.global.imm
            vimm =allfeatures_global(iter_l).imm;
            global_feature_tmp = [ global_feature_tmp, vimm*w_imm1];
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

    c = 0;
    % for iter_l = 4+imginit:imgend
    % length(allfeatures_local)
    % imgend
    % for iter_l = length(allfeatures_local)-imgend+1: length(allfeatures_local)-imginit +1 %25
    % for iter_l = 49-imgend+1: 49-imginit +1 %25
    for iter_l = 4 + imginit : 4+imgend

        global_feature_tmp = [];
        local_feature_tmp = [];
        distintic_feature_tmp = [];
	    if para.local.bsp
	    	vbsp = allfeatures_local(iter_l).dscr_bsp ;
	        local_feature_tmp = [ local_feature_tmp, vbsp*w_bsp2];
	    end
        if para.local.finddd
            vfinddd = allfeatures_local(iter_l).dscr_finddd ;
            local_feature_tmp = [ local_feature_tmp, vfinddd*w_finddd2];
        end
        if para.local.sc
            vsc = allfeatures_local(iter_l).dscr_sc ;
            local_feature_tmp = [ local_feature_tmp, vsc*w_sc2];
        end
        if para.local.dlcm
            vdlcm = allfeatures_local(iter_l).dscr_dlcm ;
            local_feature_tmp = [ local_feature_tmp, vdlcm*w_dlcm2];
        end
        if para.local.sift
            vsift = allfeatures_local(iter_l).dscr_sift ;
            local_feature_tmp = [ local_feature_tmp, vsift*w_sift2];
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
        if para.global.dlcm
            vdlcm =allfeatures_global(iter_l).dlcm;
            global_feature_tmp = [ global_feature_tmp, vdlcm*w_dlcmb2];
        end
        if para.global.imm
            vimm =allfeatures_global(iter_l).imm;
            global_feature_tmp = [ global_feature_tmp, vimm*w_imm2];
        end
        if para.distintic.size
            distintic_feature_tmp = [ distintic_feature_tmp, allfeatures_distintic.size2d(iter_l)*w_size2];
        end
        local_feature = [ local_feature, local_feature_tmp ];
        global_feature = [ global_feature, global_feature_tmp ];
        distintic_feature = [ distintic_feature, distintic_feature_tmp ];

        frame_instance = [local_feature_tmp, global_feature_tmp, distintic_feature_tmp  ];
        video_instance = [video_instance; frame_instance];

        % if iter_l > 49-imgend
        %     feature_acumulative = feature_acumulative + frame_instance;
        % end

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
