% warning off
% clear all
% close all
% clc

flag = true;
 
para.bsp = 0;
para.finddd = 0;
para.lbp = 0;
para.sc = 0;
para.dlcm = 0;
para.sift = 1;
is_norm = 1;

addpath('./BSplineFitting');
addpath('./LLC');
addpath('./SurfaceFeature');
addpath('./Functions');
addpath('./FINDDD');
addpath(genpath([pwd,'/GPML']));
addpath('./ShapeContent');
addpath('./Utilities');
addpath('./vlfeat/toolbox');
vl_setup
startup

%% script setting
% the file is start with date to distinguish
flile_header = 'clothes_dataset_RH';
%create firectory
dataset_dir = ['~/',flile_header];
current_dir='~/bags';

category = {'towel','pant','shirt','tshirt','sweater'};
% category = {'pant'};
size_class=3;
size_move=30;

% clothes is the number of flattening experiments, n_iteration is the
% number of flattening iteration in each experiment [1:7,10:12,15:16]
clothes = [1:50];
captures = 0:20;
kofkmeans = 256;
coding_opt = 'LLC'
pooling_opt = 'sum'
knn = 5

knnfinddd=100;
knnsc =200;
knnsift =400;

%% read code book 
codebook_dir = [current_dir,'/Features/'];
load([codebook_dir,'code_book',num2str(kofkmeans),'.mat']);

%% main loop
for iter_i = 1:length(category)
for iter_j = 1:size_class
for iter_k = 1:size_move

% iter_i = 1;
% iter_j = 1;
% iter_k = 20;

name_file = [current_dir,'/Features/local_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k)]
if exist([name_file '.mat'],'file')
load([name_file '.mat']);
    
for iter_it = 1:length(allfeatures_local)
    local_descriptors = allfeatures_local(iter_it);
% for iter_it = 1:length(clothes)
%     clothes_i = clothes(iter_it);
%     disp(['start read descriptors of clothes id: ', num2str(clothes_i), ' ...']);
    
%     if clothes_i < 10
%         current_dir = strcat(dataset_dir,'/0',num2str(clothes_i),'/');
%     else
%         current_dir = strcat(dataset_dir,'/',num2str(clothes_i),'/');
%     end
    
%     % feature extraction
%     for iter_j = 1:length(captures)
%         capture_i = captures(iter_j);
%         % read features from the disk
%         featureFile = strcat(current_dir,'Features/local_descriptors_capture',num2str(capture_i),'.mat');
        
%         if ~exist(featureFile,'file')
%             continue;
%         end
        
%         load(featureFile);
        
        %% coding
        if para.bsp
            if strcmp(coding_opt,'BOW')
                [ code.bsp ] = Coding( local_descriptors.bsp, code_book.bsp, is_norm );
            end
            if strcmp(coding_opt,'LLC')
                code.bsp = LLC_pooling( local_descriptors.bsp, code_book.bsp, code_book.bsp_weights, knn, pooling_opt );
            end
            allfeatures_local(iter_it).dscr_bsp = code.bsp;
        end
        if para.finddd
            vector_weights = code_book.bsp_weights;
            codebookfindd = code_book.finddd;
            if knnfinddd < kofkmeans
                vector_weights = code_book.bsp_weights(1:knnfinddd);
                codebookfindd = code_book.finddd(1:knnfinddd,:);
            end
            if strcmp(coding_opt,'BOW')
                [ code.finddd ] = Coding( local_descriptors.finddd, code_book.finddd, is_norm );
            end
            if strcmp(coding_opt,'LLC')
                code.finddd = LLC_pooling( local_descriptors.finddd, codebookfindd, vector_weights, knn, pooling_opt );
            end
            allfeatures_local(iter_it).dscr_finddd2 = code.finddd;
        end
        if para.lbp
            if strcmp(coding_opt,'BOW')
                [ code.lbp ] = Coding( local_descriptors.lbp, code_book.lbp, is_norm );
            end
            if strcmp(coding_opt,'LLC')
                code.lbp = LLC_pooling( local_descriptors.lbp, code_book.lbp,code_book.bsp_weights, knn, pooling_opt );
            end
            allfeatures_local(iter_it).dscr_lbp = code.lbp;
        end
        if para.sc
            vector_weights = code_book.bsp_weights;
            codebooksc = code_book.sc;
            if knnsc < kofkmeans
                vector_weights = code_book.bsp_weights(1:knnsc);
                codebooksc = code_book.sc(1:knnsc,:);
            end
            if strcmp(coding_opt,'BOW')
                [ code.sc ] = Coding( local_descriptors.sc, code_book.sc, is_norm );
            end
            if strcmp(coding_opt,'LLC')
                code.sc = LLC_pooling( local_descriptors.sc, codebooksc, vector_weights, knn, pooling_opt );
            end
            allfeatures_local(iter_it).dscr_sc2 = code.sc;
        end
        if para.dlcm
            if strcmp(coding_opt,'BOW')
                [ code.dlcm ] = Coding( local_descriptors.dlcm, code_book.dlcm, is_norm );
            end
            if strcmp(coding_opt,'LLC')
                code.dlcm = LLC_pooling( local_descriptors.dlcm, code_book.dlcm, code_book.bsp_weights, knn, pooling_opt );
            end
            allfeatures_local(iter_it).dscr_dlcm = code.dlcm;
        end
        if para.sift
            vector_weights = code_book.bsp_weights;
            codebooksift = code_book.sift;
            if knnsift > kofkmeans
                vector_weights = [code_book.bsp_weights code_book.bsp_weights(1:(knnsift-kofkmeans))];
                codebooksift = [code_book.sift ; code_book.sift(1:(knnsift-kofkmeans),:)];
            end
            if strcmp(coding_opt,'BOW')
                [ code.sift ] = Coding( local_descriptors.sift, code_book.sift, is_norm );
            end
            if strcmp(coding_opt,'LLC')
                code.sift = LLC_pooling( local_descriptors.sift, codebooksift, vector_weights, knn, pooling_opt );
            end
            allfeatures_local(iter_it).dscr_sift2 = code.sift;
        end
        
        % code_dir = [ current_dir, 'Codes' ];
        % if ~exist(code_dir,'dir')
        %     mkdir(code_dir);
        % end
              
        % save([code_dir,'/',coding_opt,'_codes_capture',num2str(capture_i),'.mat'],'code')
        
        % clear code;
end
            save([current_dir,'/Features/local_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'],'allfeatures_local');

end
    %%
    % disp(['fininsh coding of clothing ', num2str(clothes_i), ' ...']);
end
end
end