
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
coding_opt = 'LLC'
para.isnorm = 1

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

para.distintic.keyparthist = 0;
para.distintic.keyparthist_onlyneck = 0;
para.distintic.keyparthist_onlywaist = 0;
para.distintic.neckshirt = 0;
para.distintic.size = 0;

% the file is start with date to distinguish
flile_header = 'processed_data';%'ProcessedData';
%create firectory
% dataset_dir = ['~/clothes_dataset_RH/',flile_header];
dataset_dir = '~/clothes_dataset_RH';

% clothes is the number of flattening experiments, n_iteration is the
% number of flattening iteration in each experiment [1:7,10:12,15:16];
clothes = [1:50];

% test 1 [2:5, 7:9, 11, 13:20, 22:50]
cat1 = [ 1     2     3     4     5    27    28    29    30    45 ];
cat2 = [ 6     7     8     9    22    23    32    33    34    35 ];
cat3 = [ 10    11    13    17    18    46    47    48    49    50 ];
cat4 = [ 12    14    15    16    19    20    26    31    36    37 ];
cat5 = [ 21    24    25    38    39    40    41    42    43    44 ];


captures = 0:20;

%Read distintic features
file_dist_path = '/home/koul/PreCloCaMa/clothes_recognitionMATLAB/data/distintic_feature.mat';
load(file_dist_path);


%% main loop

Instance = [];
Label = [];
ClothesID = [];
collecton_video = [];
id_colecction = 1;

for iter_i = 1:length(clothes)
    clothes_i = clothes(iter_i);
    disp(['start read descriptors of clothes id: ', num2str(clothes_i), ' ...']);
    
    if clothes_i < 10
        current_dir = strcat(dataset_dir,'/0',num2str(clothes_i),'/');
    else
        current_dir = strcat(dataset_dir,'/',num2str(clothes_i),'/');
    end
    
    % read the label information
    labelFile = strcat(current_dir,'info.mat');
    load(labelFile);
    
    switch category
        case 't-shirt'
            label = 1;
        case 'shirt'
            label = 2;
        case 'thick-sweater'
            label = 3;
        case 'jean'
            label = 4;
        case 'towel'
            label = 5;
        otherwise
            pause        
    end
    
    % feature extraction
    for iter_j = 1:length(captures)
        
        capture_i = captures(iter_j);
        
        local_feature = [];
        local_feature_simple = [];
        global_feature = [];
        global_feature_simple = [];

        %% read features from the disk
        % read local features (code)
        % localFeatureFile = strcat(current_dir,'Features/',coding_opt,'_codes_capture',num2str(capture_i),'.mat');
        localFeatureFile = strcat(current_dir,'Codes/',coding_opt,'_codes_capture',num2str(capture_i),'.mat');
        
        if ~exist(localFeatureFile,'file')
            continue;
        else
            load(localFeatureFile);
        end
        
        factor=4;

        if para.local.bsp
            local_feature = [ local_feature, code.bsp ];
            %B = code.bsp (1:factor:end);
            %B = resample ( code.bsp ,1:length(code.bsp), factor); %% MEJORA =D
            B = resample ( code.bsp , 4, 5,1,0);
            E = -sum(code.bsp.*log2(code.bsp));
            B =  code.bsp(1:125) + code.bsp(126:250); 
            %local_feature_simple = [ local_feature_simple, [ sum(code.bsp) min(code.bsp) max(code.bsp) mean(code.bsp) median(code.bsp) std(code.bsp)  var(code.bsp) ] ];
            local_feature_simple = [ local_feature_simple, code.bsp ];
            % disp('bsp')
            % size(code.bsp)
            % size(B)
        end
        if para.local.finddd
            local_feature = [ local_feature, code.finddd ];
        end
        if para.local.lbp
            local_feature = [ local_feature, code.lbp ];
        end
        if para.local.sc
            local_feature = [ local_feature, code.sc ];
        end
        if para.local.dlcm
            local_feature = [ local_feature, code.dlcm ];
        end
        if para.local.sift
            local_feature = [ local_feature, code.sift ];
        end
        
        % read global features
        globalFeatureFile = strcat(current_dir,'Features/global_descriptors_capture',num2str(capture_i),'.mat');
        load(globalFeatureFile);
        
        if para.global.lbp
            global_feature = [ global_feature, global_descriptors.lbp ];
            %B = global_descriptors.lbp (1:factor:end);
            %B = resample ( global_descriptors.lbp , 1:length(global_descriptors.lbp), factor); mejora!!
            %B = resample ( global_descriptors.lbp , 2, 3);    mantiene
            [idx,C] = kmeans(reshape (global_descriptors.lbp,length(global_descriptors.lbp),1), factor); 
            B = reshape ( C , 1, factor);  
            E = -sum(global_descriptors.lbp.*log2(global_descriptors.lbp));
            %global_feature_simple = [ global_feature_simple, [ sum(global_descriptors.lbp) min(global_descriptors.lbp) max(global_descriptors.lbp) mean(global_descriptors.lbp) median(global_descriptors.lbp) std(global_descriptors.lbp) var(global_descriptors.lbp) ] ];
            global_feature_simple = [ global_feature_simple,  global_descriptors.lbp];
            % disp('lbp')
            % size(global_descriptors.lbp)
            % size(B)
        end
        if para.global.si
            global_feature = [ global_feature, global_descriptors.si ];
            %B = global_descriptors.si (1:factor:end);
            %B = resample ( global_descriptors.si , 1:length(global_descriptors.si), factor);
            B = resample ( global_descriptors.si , 2, 6);
            E = -sum(global_descriptors.si.*log2(global_descriptors.si));
            %global_feature_simple = [ global_feature_simple, [ sum(global_descriptors.si) min(global_descriptors.si) max(global_descriptors.si) mean(global_descriptors.si) median(global_descriptors.si) std(global_descriptors.si) var(global_descriptors.si) ] ];
             global_feature_simple = [ global_feature_simple, global_descriptors.si  ];
            % disp('si')
            % size(global_descriptors.si)
            % size(B)
        end
        if para.global.topo
            global_feature = [ global_feature, global_descriptors.topo ];
            %B = global_descriptors.topo (1:factor:end);
            %B = resample ( global_descriptors.topo , 1:length(global_descriptors.topo), factor);
            B = resample ( global_descriptors.topo , 1, 3);
            E = -sum(global_descriptors.topo.*log2(global_descriptors.topo));
            %global_feature_simple = [ global_feature_simple, [ sum(global_descriptors.topo) min(global_descriptors.topo) max(global_descriptors.topo) mean(global_descriptors.topo) median(global_descriptors.topo) std(global_descriptors.topo) var(global_descriptors.topo)  ] ];        
            global_feature_simple = [ global_feature_simple,  global_descriptors.topo ];
            % disp('topo')
            % size(global_descriptors.topo)
            % size(B)
        end
        if para.global.dlcm
            global_feature = [ global_feature, global_descriptors.dlcm];
             B = global_descriptors.dlcm (1:factor:end);
            E = -sum(global_descriptors.dlcm.*log2(global_descriptors.dlcm));
            %global_feature_simple = [ global_feature_simple, [ sum(global_descriptors.dlcm) min(global_descriptors.dlcm) max(global_descriptors.dlcm) mean(global_descriptors.dlcm) median(global_descriptors.dlcm) std(global_descriptors.dlcm) var(global_descriptors.dlcm)  ] ];        
            global_feature_simple = [ global_feature_simple, B ];
        end
        if para.global.imm
            global_feature = [ global_feature, global_descriptors.imm];
             B = global_descriptors.imm (1:factor:end);
            E = -sum(global_descriptors.imm.*log2(global_descriptors.imm));
            %global_feature_simple = [ global_feature_simple, [ sum(global_descriptors.imm) min(global_descriptors.imm) max(global_descriptors.imm) mean(global_descriptors.imm) median(global_descriptors.imm) std(global_descriptors.imm)  var(global_descriptors.imm)  ] ];        
            global_feature_simple = [ global_feature_simple, B ];
        end
        if para.global.vol
            global_feature = [ global_feature, global_descriptors.vol];
            B = global_descriptors.vol (1:factor:end);
            E = -sum(global_descriptors.vol.*log2(global_descriptors.vol));
            %global_feature_simple = [ global_feature_simple, [ sum(global_descriptors.vol) min(global_descriptors.vol) max(global_descriptors.vol) mean(global_descriptors.vol) median(global_descriptors.vol) std(global_descriptors.vol) var(global_descriptors.vol)  ] ];
            global_feature_simple = [ global_feature_simple, B ];
        end        
        
       instance = [ local_feature, global_feature ];
        % instance = [ local_feature_simple, global_feature_simple ];
        
        Instance = [ Instance; instance ];
        Label = [ Label; label ];
        ClothesID = [ ClothesID; clothes_i ];

        collecton_video (id_colecction,:,:) = instance;
        id_colecction = id_colecction+1;
        clear instance;
    end
    %%
    disp(['fininsh reading of clothing ', num2str(clothes_i), ' ...']);
    clear label1 label2;
end

if para.distintic.neckshirt
    Instance = [ Instance, distintic_feature.shirtneck ];
end
if para.distintic.size
    Instance = [ Instance, distintic_feature.size2d ];
end
if para.distintic.keyparthist
    Instance = [ Instance, distintic_feature.keyparthist ];
end
if para.distintic.keyparthist_onlyneck
    Instance = [ Instance, distintic_feature.keyparthist_onlyneck ];
end
if para.distintic.keyparthist_onlywaist
    Instance = [ Instance, distintic_feature.keyparthist_onlywaist ];
end



%% generate model for robot practice
if para.isnorm
    [ Instance Label norm ] = prepareData( Instance, Label );
else
    norm = [];
end

% clearvars -except Instance Label ClothesID norm; 


%% traning model for robot practical recognition
%train SVM model
svm_opt = '-s 0 -c 10 -t 2 -g 0.01';

svm_struct = libsvmtrain( Label, Instance, svm_opt );
save('classifier_svm.mat','svm_opt','norm','svm_struct');
% % %%

%% training GP
% % kernel = @covSEiso;
% % para.kernel = kernel;
% % para.hyp = log([ones(1,1)*46, 11]);
% % para.S = 1e4;
% % labels = unique(Label);
% % c = length(labels);
% % para.c = c;
% % para.Ncore = 12;
% % para.flag = true; 
% % hyp = para.hyp;
% % gp_para = para;
% % 
% % % estimate the posterior probility of p(f|X,Y)
% % [ K ] = covMultiClass(hyp, para, Instance, []);
% % [ gp_model ] = LaplaceApproximation(hyp, para, K, Instance, Label);
% % save('classifier_gp_demo.mat','gp_model','norm','gp_para');

%%

% % % % % % training random forest
rf_opt.treeNum = 1000;
rf_opt.mtry = 200;
para.opt = rf_opt;
% % rf_struct = classRF_train( Instance, Label, rf_opt.treeNum, rf_opt.mtry );
% % save('classifer.mat', 'rf_opt', 'rf_struct');

    
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

[ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'SVM', para );
% [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'adaboost', para );
% [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'NN', para );
% [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'fuzzy', para );
% [ result_svm ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'RF', para );


disp('press Enter to continue ...');
pause
close all

%%  set Gaussian Process parameters

kernel = @covLINiso;

fold = 2;
expNum = 1;
isnorm = 1;

para.kernel = kernel;

para.hyp = log([11]);

para.model_selection = 1;
para.sampe_rate = 0.3;

para.labels = labels;
para.fold = fold;
para.isnorm = isnorm;
para.S = 1e4;
para.cv_mode = 'clothes';
%%


%[ result ] = x_fold_CV( Instance, Label, ClothesID, fold, expNum, 'myGP', para );


% % for expi = 1:1
% %     [ result ] = LeaveOneOutValidification(Instance, Label, ClothesID, 'SVM', para );
% % end

