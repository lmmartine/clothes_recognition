
warning off
%pctRunOnAll warning('off','all')
clear all
close all
clc

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
para.isnorm = 1

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

current_dir='/home/lmartinez/bags';
data_dir = '/home/lmartinez/bags/data/';


% category = {'pant','shirt'};
category = {'pant','shirt','tshirt','sweater'};
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
for iter_k = 5:size_move

name_local_file = [current_dir,'/Features/local_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)]
name_global_file = [current_dir,'/Features/global_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)]
name_distintic_file = [current_dir,'/Features/distintic_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)]
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


    global_feature_tmp = [];
    local_feature_tmp = [];
    distintic_feature_tmp = [];

    if para.local.bsp
        local_feature_tmp = [ local_feature_tmp, allfeatures_local(1).dscr_bsp ];
    end
    if para.global.lbp
        global_feature_tmp = [ global_feature_tmp, allfeatures_global(1).lbp ]; %*256/174
    end
    if para.global.si
        global_feature_tmp = [ global_feature_tmp, allfeatures_global(1).si ]; % *256/9 
    end
    if para.global.topo
        global_feature_tmp = [ global_feature_tmp, allfeatures_global(1).topo ]; % *256/100
    end
    if para.distintic.size
        distintic_feature_tmp = [ distintic_feature_tmp, allfeatures_distintic.size2d(1) ]; % /(640*480)
    end

    local_feature = [ local_feature, local_feature_tmp ];
    global_feature = [ global_feature, global_feature_tmp ];
    distintic_feature = [ distintic_feature, distintic_feature_tmp ];

    frame_instance = [local_feature_tmp, global_feature_tmp, distintic_feature_tmp ];
    video_instance = [video_instance; frame_instance];


    % for iter_l = 5:min(length(allfeatures_local),9)
    for iter_l = length(allfeatures_local)-25: length(allfeatures_local) %25
    % for iter_l = max(5,length(allfeatures_local)-20): length(allfeatures_local)
        global_feature_tmp = [];
        local_feature_tmp = [];
        distintic_feature_tmp = [];

        if para.local.bsp
            local_feature_tmp = [ local_feature_tmp, allfeatures_local(iter_l).dscr_bsp ];
        end
        if para.global.lbp
            global_feature_tmp = [ global_feature_tmp, allfeatures_global(iter_l).lbp];
        end
        if para.global.si
            global_feature_tmp = [ global_feature_tmp, allfeatures_global(iter_l).si  ];
        end
        if para.global.topo
            global_feature_tmp = [ global_feature_tmp, allfeatures_global(iter_l).topo ];
        end
        if para.distintic.size
            distintic_feature_tmp = [ distintic_feature_tmp, allfeatures_distintic.size2d(iter_l)];
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
    [name_local_file '.mat or ' name_global_file '.mat doesnt exist'] 
end

end
    id = id+1;
end
end






% %% generate model for robot practice
% if para.isnorm
%     [ Instance Label2 norm ] = prepareData( Instance, Label );
% else
%     norm = [];
% end

% % clearvars -except Instance Label ClothesID norm; 


% %% traning model for robot practical recognition
% %train SVM model
svm_opt = '-s 0 -c 10 -t 2 -g 0.01';

% % % svm_struct = libsvmtrain( Label, Instance, svm_opt );
% % % save('classifier_demo.mat','svm_opt','norm','svm_struct');
% % % %%

% %% training GP
% % % kernel = @covSEiso;
% % % para.kernel = kernel;
% % % para.hyp = log([ones(1,1)*46, 11]);
% % % para.S = 1e4;
% % % labels = unique(Label);
% % % c = length(labels);
% % % para.c = c;
% % % para.Ncore = 12;
% % % para.flag = true; 
% % % hyp = para.hyp;
% % % gp_para = para;
% % % 
% % % % estimate the posterior probility of p(f|X,Y)
% % % [ K ] = covMultiClass(hyp, para, Instance, []);
% % % [ gp_model ] = LaplaceApproximation(hyp, para, K, Instance, Label);
% % % save('classifier_gp_demo.mat','gp_model','norm','gp_para');

% %%

% % % % % % % training random forest
% rf_opt.treeNum = 1000;
% rf_opt.mtry = 200;
% para.opt = rf_opt;
% % % rf_struct = classRF_train( Instance, Label, rf_opt.treeNum, rf_opt.mtry );
% % % save('classifer.mat', 'rf_opt', 'rf_struct');

    
%% classfication varification
fold = 5;
expNum = 10;
para.opt = svm_opt;
para.cv_mode = 'clothes';
labels = unique(Label);
c = length(labels);
para.c = c;
para.Ncore = 12;
para.flag = true; 


% [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'SVM', para );
% % [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'adaboost', para );
% % [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'NN', para );
% % [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'fuzzy', para );
% % [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'RF', para );

% % div = 
% % training_inst = Instance(trainIndex,:);
% % training_label = Label(trainIndex);
% % testing_inst = Instance(testIndex,:);
% % testing_label = Label(testIndex);

% % if strcmp(classifier,'SVM')
% %     opt = para.opt;
% %     model = libsvmtrain( training_label, training_inst, opt );
% %     [predict_label, accuracy, dec_values] = libsvmpredict( testing_label, testing_inst, model);
% %     prob = dec_values;
% % end


% disp('press Enter to continue ...');
% pause
% close all
