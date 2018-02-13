function video_features = c2_extract_features(video)

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
para.sensor = 'kinect'; 

%% feature setting
para.local.bsp = 1;
para.local.finddd = 0;
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

para.abheight = 0;
para.iscontinue = 0;
flag = 0; % change it to 1 for visualization


se = strel('diamond',3);

size_xtion1 = 35;
size_xtion2 = 4;


	infposition = video.InfLabel;
	

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
	while iter_a <= 4%length(video.ImgDepth) %&& length(allfeatures_local(:).dscr_bsp) < 38
		if last_iter_a == iter_a %|| iter_a==38|| iter_a==39
			iter_a = iter_a+1
			continue
		end
		last_iter_a = iter_a;
		% iter_a
		imagedepth=readImage(video.ImgDepth(iter_a));
		imagemask=readImage(video.ImgMask(iter_a));
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

video_features.global = allfeatures_global;
video_features.local = allfeatures_local;
video_features.dist = allfeatures_distintic;
% end


