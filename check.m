current_dir='~/bags';
data_dir = '~/bags/data/';


% category = {'shirt'};
% category = {'pant','tshirt','sweater','towel'};
category = {'pant','shirt','tshirt','sweater','towel'};

size_class=3;
size_move=10;

xtion1 =[];
xtion2= [];

%% main loop
nmax = 0;
for iter_i = 1:length(category)
for iter_j = 1:size_class
% iter_i=1;
% iter_j=3;
	nclass = 0 ;
for iter_k = 1:size_move

name_id_file = [current_dir,'/Features/id_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'];

name_local_file = [current_dir,'/Features/local_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'];
name_global_file = [current_dir,'/Features/global_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'];
name_dist_file = [current_dir,'/Features/distintic_descriptors_' category{iter_i} int2str(iter_j) '_move' int2str(iter_k) '.mat'];


if exist([name_id_file ],'file')
load([name_id_file ]);

			infposition = allfeatures_id;

			% n_xtion2 = sum(str2double(infposition) < 100);
			n_xtion2 = length(infposition) - 4;

			if n_xtion2 <55
				disp([category{iter_i} int2str(iter_j) '_move' int2str(iter_k) ' : ' int2str(n_xtion2)  ])
				% delete (name_id_file, name_local_file, name_global_file, name_dist_file)
			% else
			% 	nclass = nclass +1;
			% 	xtion2 = [xtion2 ; n_xtion2];
			% % disp([category{iter_i}, ' ', int2str(iter_j) ' ' int2str(iter_k), ' : ', num2str(n_xtion2)])
			end	

			if n_xtion2 > nmax
			 nmax = 	n_xtion2;
			 end	

end

end
% nclass
end
end

nmax
