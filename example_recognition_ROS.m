
% Requirement : rosinit

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


subrgb = rossubscriber('/camera/rgb/image_raw');
subdepth = rossubscriber('/camera/depth/image_raw');

imagergb = readImage(receive(subrgb,10));
imagedepth = readImage(receive(subdepth,10));

imshow(imagergb)

%Extract features
rangeMap = double(imagedepth*0.1); 
% rangeMap = imresize(rangeMap, imresizeScale);

para.mask = mask;
[ model ] = SurfaceAnalysis( rangeMap, para, flag );
