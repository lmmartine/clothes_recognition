
flag = true;
imresizeScale = 0.2;

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
addpath(genpath([pwd,'/geodesic']));
addpath('./ClothesUtilities');
addpath('./FINDDD');
addpath('./SimpleFeatures');
addpath('./3DVol');

vl_setup
startup


%% sensor setting
para.sensor = 'kinect';  % RH or RH_fast or kinect

%% feature setting
para.local.bsp = 1;
para.local.finddd = 0;
para.local.lbp = 1;
para.local.sc = 0;
para.local.dlcm = 0;
para.local.sift = 0;

para.global.si = 1;
para.global.lbp = 1;
para.global.topo = 1;
para.global.dlcm = 0;
para.global.imm = 0;
para.global.vol = 0;

para.abheight = 0;
para.iscontinue = 0;
flag = 0; % change it to 1 for visualization

current_dir='~/bags';
data_dir = '~/bags/data/';
if ~exist([current_dir,'/Features'],'dir')
    mkdir( [current_dir,'/Features'] );
end
% category = {'pant','shirt','tshirt','sweater'};
category = {'towel'};
% category = 'tshirt';
size_class=3;
size_move=10;

se = strel('diamond',3);

size_xtion1 = 35;
size_xtion2 = 4;

for iter_i = 1:length(category)
% for iter_j = 2:size_class
% 	for iter_k = 1:size_move
% 		if iter_j == 2 && iter_k ==1
% 			continue
% 		end
iter_j = 1;
iter_k = 9;
		name_file = [data_dir category{iter_i} int2str(iter_j) '_move' int2str(iter_k)]
		if exist([name_file '.mat'],'file')
			load([name_file '.mat']);
			infposition = testresp.InfLabel;
			

			% for iter_i = 1:length(infposition)
			% 	% str2double(infposition{iter_i}) < 100
			% 	if str2double(infposition{iter_i}) < 100
			% 		infposition{iter_i} = num2str(str2double(infposition{iter_i,1}) + 100);
			% 	else
			% 		infposition{iter_i} = num2str(str2double(infposition{iter_i,1}) - 100 + 1);
			% 	end
			% end


			n_xtion2 = sum(str2double(infposition) < 100);
			n_xtion1 = sum(str2double(infposition) >= 100);
			pstart_xtion1 = find(strcmp(infposition,'100'));
			allfeatures_global = [];
			allfeatures_local = [];
			allfeatures_distintic.size2d = [];
			allfeatures_id =[];
			x1=false;
			iter_a=1;  %%%%%1
			first_xtion1 = -1;
			last_iter_a =0;
			while iter_a <= length(testresp.ImgDepth) %&& length(allfeatures_local(:).dscr_bsp) < 38
				if last_iter_a == iter_a %|| iter_a==38|| iter_a==39
					iter_a = iter_a+1
					continue
				end
				last_iter_a = iter_a;
				iter_a
				imagedepth=readImage(testresp.ImgDepth(iter_a));
				imagemask=readImage(testresp.ImgMask(iter_a));
				% rangeMap = imadjust(imagedepth);%double(imagedepth*0.1); 
				rangeMap = double(imagedepth); 
	% rangeMap = double(imagedepth*0.1); 
	% rangeMap = imresize(rangeMap, imresizeScale);
				% if max(max(imagemask)) == 255
				% 	imagemask = imagemask/255;
				% end


				erodedI = imerode(imagemask,se);
				erodedI = imerode(erodedI,se);
				dilatedI = imdilate(erodedI,se);
				dilatedI = imdilate(dilatedI,se);

				dilatedI = im2bw(dilatedI, 0.5);
				% sum(sum(dilatedI));
				if sum(sum(dilatedI))>10000
					allfeatures_id = [allfeatures_id ; iter_a];
					imagemask = bwareafilt(dilatedI,1);

					if max(max(imagemask)) == 1
						imagemask = imagemask*2;
					end
					para.mask = imagemask;
					[ model ] = SurfaceAnalysis( rangeMap, para, flag );

					if para.global.si + para.global.lbp + para.global.topo + para.global.dlcm + para.global.imm + para.global.vol > 0
					    % extract local feature
					    [ global_descriptors ] = ExtractGlobalFeatures( model, para, flag );
					    allfeatures_global = [allfeatures_global; global_descriptors];
					end
					        % extract local feature
			        if para.local.bsp + para.local.finddd + para.local.lbp + para.local.sc + para.local.sift + para.local.finddd > 0
			            [ local_descriptors ] = ExtractLocalFeatures( model, para, flag );
			        	allfeatures_local = [allfeatures_local; local_descriptors];
			           
			         end
			         allfeatures_distintic.size2d = [allfeatures_distintic.size2d; sum(sum(dilatedI))];

					if str2double(infposition{iter_a}) >= 100
						% iter_a
						% (n_xtion1-1)/4 
						if first_xtion1 == -1
							first_xtion1 = iter_a - n_xtion2
						end
						iter_a = iter_a + (n_xtion1-first_xtion1)/(size_xtion1 - 1) ;
						
					elseif str2double(infposition{iter_a} )< 100
							iter_a = iter_a + (n_xtion2-1)/(size_xtion2 - 1) ;
					end
					if iter_a > pstart_xtion1 &&  x1 == false
							iter_a = pstart_xtion1;
							x1 = true;
					end
				else
					sum(sum(dilatedI))
					iter_a = iter_a+1;
				end
				iter_a = floor(iter_a) ;
			end

			%     % save features to the disk
			% if para.iscontinue  TODO
			% 	merge_descriptors([current_dir,'/Features/global_descriptors_capture',num2str(capture_i),'.mat'], global_descriptors, para, 'global');
			% else
			save([current_dir,'/Features/local_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'],'allfeatures_local');

			save([current_dir,'/Features/global_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'],'allfeatures_global');
			save([current_dir,'/Features/distintic_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'],'allfeatures_distintic');
			save([current_dir,'/Features/id_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'],'allfeatures_id');

			end

		end
% 	end
% end



