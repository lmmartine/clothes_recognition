function verify_data( category, size_class , size_move )

data_dir = '~/bags/data/';
% categories = ['pant' 'shirt' 'tshirt' 'sweater'];
% category = 'tshirt';

se = strel('diamond',3);
% for iter_i = 1:length(categories)
for iter_j = 2:size_class
	for iter_k = 1:size_move
% iter_j = 1;
% iter_k = 9 ;
		name_file = [data_dir category int2str(iter_j) '_move' int2str(iter_k)];
		if exist([name_file '.mat'],'file')
			if ~exist(name_file,'dir')
			    mkdir( name_file);
			end
			load([name_file '.mat']);

			for iter_a = 1:length(testresp.ImgDepth)

				imagedepth=readImage(testresp.ImgDepth(iter_a));
				imagemask=readImage(testresp.ImgMask(iter_a));
				rangeMap = imadjust(imagedepth);%double(imagedepth*0.1); 

				if max(max(imagemask)) == 1
					imagemask = imagemask*255;
				end

				erodedI = imerode(imagemask,se);
				erodedI = imerode(erodedI,se);
				dilatedI = imdilate(erodedI,se);
				dilatedI = imdilate(dilatedI,se);

				dilatedI = im2bw(dilatedI, 0.5);
				if sum(sum(dilatedI)) > 255*10
					dilatedI = bwareafilt(dilatedI,1);

					imwrite (rangeMap,[name_file '/imagedepth' int2str(iter_a) '.png'])
					imwrite (dilatedI,[name_file '/imagemask' int2str(iter_a) '.png'])

				end
			end 
		end
	end
end
% end


testresp.ImgDepth=testresp.ImgDepth(2:121);
testresp.ImgMask=testresp.ImgMask(2:121);
testresp.InfLabel=testresp.InfLabel(2:121);
save('~/bags/data/towel1_move9.mat','testresp');