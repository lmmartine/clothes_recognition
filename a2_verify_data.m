function verify_data( )

data_dir = '~/bags/data/';
% category = {'pant'};
category = {'pant','shirt','tshirt','sweater','towel'};

% category = 'tshirt';

size_class = 3;
size_move = 30;

se = strel('diamond',3);
for iter_i = 1:length(category)
for iter_j = 1:size_class
	for iter_k = 1:size_move
% iter_j = 1;
% iter_k = 9 ;
		name_file = [data_dir category{iter_i} int2str(iter_j) '_move' int2str(iter_k)];
		if exist([name_file '.mat'],'file')
			if ~exist(name_file,'dir')
			    mkdir( name_file);
			end
			load([name_file '.mat']);

			for iter_a = 1:length(testresp.ImgDepth)

				imgd = testresp.ImgDepth(iter_a);
				imgm = testresp.ImgMask(iter_a);
				if imgd.Height*imgd.Width == 0
					imgd = testresp.ImgDepth(iter_a-1);
					imgm = testresp.ImgMask(iter_a-1);
				end
				imagedepth=readImage(imgd);
				imagemask=readImage(imgm);

				rangeMap = imadjust(imagedepth);%double(imagedepth*0.1); 

				if max(max(imagemask)) == 1
					imagemask = imagemask*255;
				end

				erodedI = imerode(imagemask,se);
				erodedI = imerode(erodedI,se);
				dilatedI = imdilate(erodedI,se);
				dilatedI = imdilate(dilatedI,se);

				dilatedI = im2bw(dilatedI, 0.5);
				sum(sum(dilatedI))
				if sum(sum(dilatedI)) > 255*10
					dilatedI = bwareafilt(dilatedI,1);

					imwrite (rangeMap,[name_file '/imagedepth' int2str(iter_a) '.png'])
					imwrite (dilatedI,[name_file '/imagemask' int2str(iter_a) '.png'])

				end
			end 
		end
	end
end
end


% testresp.ImgDepth=testresp.ImgDepth(2:121);
% testresp.ImgMask=testresp.ImgMask(2:121);
% testresp.InfLabel=testresp.InfLabel(2:121);
% save('~/bags/data/towel1_move9.mat','testresp');