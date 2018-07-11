warning off
parpool
pctRunOnAll warning('off','all')
clear all
close all
clc

%% parameters of Robot Head data
% BSP 35 / LBP-global put 3 layer as a inst /LBP-local put 3 layer into set / DLCM-global single scale/ DLCM-local single scale
% LBP in Higfrequencypase, BSP 35 / LBP-global put 3 layer as a inst /LBP-local put 3 layer into set / DLCM-global single scale/ DLCM-local single scale
% % LBP in raw dept but 0.2 sampling_rate, BSP 35 / LBP-global put 3 layer as
% % a inst /LBP-local put 3 layer into set / DLCM-global single scale + SVD/
% % DLCM-local single scale+SVD
%%
%% parameters of Kinect data
% BSP 17 Bspline fitting 23 ZeroCrossing laplacian template 7 Shape Index filter size 3


%% sensor setting
para.sensor = 'kinect'  % RH or RH_fast or kinect

%% feature setting
para.local.bsp = 1;
para.local.finddd = 1;
para.local.lbp = 0;
para.local.sc = 1;
para.local.dlcm = 1;
para.local.sift = 1;

para.global.si = 0;
para.global.lbp = 0;
para.global.topo = 0;
para.global.dlcm = 0;
para.global.imm = 0;
para.global.vol = 0;

%% parameters
flag = 0; % change it to 1 for visualization
para.abheight = 1;
para.iscontinue = 0; % 0 for delete previous features and save new feature; 1 for overwrite old features with new features

addpath('./Functions');
addpath('./BSplineFitting');
addpath('./SurfaceFeature');
addpath('./ClothesUtilities');
addpath('./SpatialPyramid');
addpath(genpath([pwd,'/GPML']));
addpath('./ShapeContent');
addpath('./Utilities');
addpath('./vlfeat/toolbox');
addpath('./FINDDD');
addpath('./SimpleFeatures');
addpath('./3DVol');

vl_setup
startup

%% experiment setting
% the file is start with date to distinguish
flile_header = 'clothes_dataset_RH/processed_data';%'clothes_dataset_RH';
%create firectory
dataset_dir = ['~/',flile_header]; % change to your directory


% clothes is the number of flattening experiments, n_iteration is the
% number of flattening iteration in each experiment [1:7,10:12,15:16]
clothes = [1:50];%[1:50];
captures = 0:20;

%%

%% main loop

for iter_i = 1:length(clothes)
    clothes_i = clothes(iter_i);
    disp(['start read descriptors of clothes id: ', num2str(clothes_i), ' ...']);
    
    if clothes_i < 10
        current_dir = strcat(dataset_dir,'/0',num2str(clothes_i),'/');
    else
        current_dir = strcat(dataset_dir,'/',num2str(clothes_i),'/');
    end
    
    % feature extraction
    for iter_j = 1:length(captures)
        
        tic
        
        capture_i = captures(iter_j);
        % get range map of iter i
        dataFile = strcat(current_dir,'clothes_',num2str(clothes_i),'_capture_',num2str(capture_i));
        
        % Make sure the file exists (some gaps in the dataset)
        disp(strcat('loading ',dataFile,'...'));
        
        if ~exist(strcat(dataFile,'_rgb.png'),'file')
            continue
        end
        
        rgbImage = imread(strcat(dataFile,'_rgb.png')); 
        rangeMap = imread(strcat(dataFile,'_depth.png')); 
        mask = imread(strcat(dataFile, '_mask.png')); 
                
        if strcmp(para.sensor, 'RH')
            rgbImage = imresize(rgbImage, 0.5);
            rangeMap = double(rangeMap*0.1); rangeMap = imresize(rangeMap, 0.5);
        end
        if strcmp(para.sensor, 'RH_fast')
            rgbImage = imresize(rgbImage, 0.2);
            rangeMap = double(rangeMap*0.1); rangeMap = imresize(rangeMap, 0.2);
        end
        rangeMap = double(rangeMap);
        mask = imresize(imfill(mask,'hole'), size(rangeMap), 'nearest');
        
        
        %%
        % surface analysis
        para.mask = mask;
        [ model ] = SurfaceAnalysis( rangeMap, para, flag );
    
        if ~exist([current_dir,'/Features'],'dir')
            mkdir( [current_dir,'/Features'] );
        end
        
        if para.global.si + para.global.lbp + para.global.topo + para.global.dlcm + para.global.imm + para.global.vol > 0
            % extract local feature
            [ global_descriptors ] = ExtractGlobalFeatures( model, para, flag );
        
            % save features to the disk
            if para.iscontinue
                merge_descriptors([current_dir,'/Features/global_descriptors_capture',num2str(capture_i),'.mat'], global_descriptors, para, 'global');
            else
                save([current_dir,'/Features/global_descriptors_capture',num2str(capture_i),'.mat'],'global_descriptors');
            end
        end
        
        % extract local feature
        if para.local.bsp + para.local.finddd + para.local.lbp + para.local.sc + para.local.sift + para.local.finddd > 0
            [ local_descriptors] = ExtractLocalFeatures( model, para, flag );
        
            % save features to the disk
            if para.iscontinue
                merge_descriptors([current_dir,'/Features/local_descriptors_capture',num2str(capture_i),'.mat'], local_descriptors, para, 'local');
            else
                save([current_dir,'/Features/local_descriptors_capture',num2str(capture_i),'.mat'],'local_descriptors');
            end
         end
        
        clear local_descriptors global_descriptors;
        
        toc
                
        pause(1)
        close all
    end
    %%
    disp(['fininsh feature extraction of clothes ', num2str(iter_i), ' ...']);
end
