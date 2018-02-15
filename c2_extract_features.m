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
first_xtion1 = -1;
last_iter_a =0;

% length(video.ImgDepth)
index_vector = [1 length(video.ImgDepth)-5:length(video.ImgDepth)];
% index_vector

for iter_a = index_vector%length(video.ImgDepth) %&& length(allfeatures_local(:).dscr_bsp) < 38
	% iter_a
	imagedepth=readImage(video.ImgDepth(iter_a));
	imagemask=readImage(video.ImgMask(iter_a));
	rangeMap = double(imagedepth); 

	erodedI = imerode(imagemask,se);
	erodedI = imerode(erodedI,se);
	dilatedI = imdilate(erodedI,se);
	dilatedI = imdilate(dilatedI,se);

	dilatedI = im2bw(dilatedI, 0.5);
	% sum(sum(dilatedI))

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

end

video_features.global = allfeatures_global;
video_features.local = allfeatures_local;
video_features.dist = allfeatures_distintic;