function [best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, type_distance, nexp)
tic
% nCollection=17;
% nC=17;
addpath('./LLC');
addpath('./drtoolbox');
addpath('./drtoolbox/techniques');

%% read code book 
current_dir='~/bags';
codebook_dir = [current_dir,'/Features/'];
kofkmeans = 256;
pooling_opt = 'sum';
knn = 10;
load([codebook_dir,'code_book',num2str(kofkmeans),'.mat']);

[nC, collection_frames, feats] = size(collecton_video);

collecton_video_unido = [];
for i=1:nC
	% n = floor(i/30) + 1;
	X = reshape(collecton_video(i,:,:), [collection_frames feats]);
	% X = [ones(collection_frames, 1)*n X];
	collecton_video_unido = [collecton_video_unido; X];
end

% LLC al conjunto de caracteristicas del video, reduciendolo a 10, es el mejor resultado encontrado
collecton_video_llc=[];
[nC, collection_frames, feats] = size(collecton_video);
% for i=1:nC
	% i
	% X = reshape(collecton_video(i,:,:), [collection_frames feats]);
	 % collecton_video_bsp = collecton_video_unido(:,1:256);
	 % collecton_video_finddd = collecton_video_unido(:,256+1:256*2);
	% collecton_video_sift = collecton_video_unido(:,256*4+1:256*5);
	collecton_video_dlcm = collecton_video_unido(:,256*3+1:256*4);
	collecton_video_sc = collecton_video_unido(:,256*2+1:256*3); %65

	% [mappedXllc_bsp, mapping] = compute_mapping(collecton_video_bsp, 'GPLVM',nout);
	% [mappedXllc_findd, mapping] = compute_mapping(collecton_video_finddd, 'GPLVM',nout);
	% [mappedXllc_sift, mapping] = compute_mapping(collecton_video_sift, 'GPLVM',nout);
	[mappedXllc_dlcm, mapping] = compute_mapping(collecton_video_dlcm, 'GPLVM',100);
	[mappedXllc_sc, mapping] = compute_mapping(collecton_video_sc, 'GPLVM', 200);
	% size(X)
	% [mappedXllc, mapping] = compute_mapping(X, 'LMNN');
	% mappedXllc= [mappedXpca; mappedXllc] ;
	% mappedXllc = [collecton_video_unido(:,[1:256*3 256*4+1:feats]) mappedXllc];
	% mappedXllc = [collecton_video_unido(:,[1:256*2 256*3+1:feats]) mappedXllc]; %65
	mappedXllc = [collecton_video_unido(:,[1:256*2 256*4+1:feats]) mappedXllc_sc mappedXllc_dlcm ];


	% save([current_dir,'/mappedXllc.mat'],'mappedXllc');
	[size1, size2] = size(mappedXllc);
for i=1:nC
	X= mappedXllc(collection_frames*(i-1)+1:collection_frames*(i),:);
	% size(X)
	mapped = reshape(X, [1 collection_frames size2]);
	collecton_video_llc(i,:,:) = mapped;
end
	
	
save([current_dir,'/collecton_video_llc.mat'],'collecton_video_llc');
% load([current_dir,'/collecton_video_llc.mat']);

collecton_video = collecton_video_llc;



n_best = 5;

% [nC, collection_frames, feats] = size(collecton_video);
n_class = length(unique(Label));
conf_matrix = zeros(n_class,n_class);
conf_matrix2 = zeros(n_class,n_class);
	
collecton_video_train = collecton_video(vector_train,:,:) ;
[nC2, collection_frames, feats] = size(collecton_video_train);
Label_sub = Label(vector_train,:,:) ;
result = zeros(length(vector_test),1);
result2 = zeros(length(vector_test),1);



parfor (id=1:length(vector_test),3)
% for id=1:length(vector_test)
% for id=1:min(20,length(vector_test))
% id=1;
	i = vector_test(id);
	id_c = Label (int8(i));
	Y = collecton_video(i,[1 collection_frames-n_best+1:collection_frames],:);
	% Y = collecton_video(i,[1 6:10],:);
	collecton_video_train_best = zeros(nC2, n_best+1, feats);
	
	for c=1:nC2
		collecton_video_train_best(c,1,:) = collecton_video_train(c,1,:);
		% matrix_distance = zeros(n_best+1, collection_frames);
		ids = [];
		for fy=2:(n_best+1)
			dc = zeros(collection_frames, 1);
			v1=Y(1, fy,:);
			v1 = reshape(v1,[1 length(v1)]);
			v1 = real (v1);
			for ft=2:collection_frames
				v2=collecton_video_train(c,ft,:);
				v2 = reshape(v2,[1 length(v2)]);
				v2 = real (v2);
				dc(ft) = pdist([v1; v2], type_distance); %best result
				% dc(ft) = norm([v1; v2], 'inf');
				% dc(ft) = sum(abs((v1- v2).^5)).^(1/5); 
				% matrix_distance(fy,ft) = dc(ft);
			end
			% best
			dc(1) = max(dc);
			% for i=1:length(ids)
			% 	dc(ids(i)) = max(dc);
			% end
			[a,idmin]= min(dc);
			% dc(idmin) = max(dc);
			% [a,idmin]= min(dc);
			ids = [ids idmin];
			[B,I] = sort(dc);
			collecton_video_train_best(c,fy,:) = collecton_video_train(c,idmin,:);
		end
		% matrix_distance
	end 


	% % % WITH MATRIX DISTANCE
	% for c=1:nC2
	% 	collecton_video_train_best(c,1,:) = collecton_video_train(c,1,:);
	% 	matrix_distance = zeros(n_best+1, collection_frames);
	% 	ids = [];
	% 	for fy=2:(n_best+1)
	% 		dc = zeros(collection_frames, 1);
	% 		v1=Y(1, fy,:);
	% 		v1 = reshape(v1,[1 length(v1)]);
	% 		v1 = real (v1);
	% 		for ft=2:collection_frames
	% 			v2=collecton_video_train(c,ft,:);
	% 			v2 = reshape(v2,[1 length(v2)]);
	% 			v2 = real (v2);
	% 			% dc(ft) = pdist([v1; v2], type_distance);
	% 			% dc(ft) = norm([v1; v2], 'inf');
	% 			dc(ft) = sum(abs((v1- v2).^3)).^(1/5); 
	% 			matrix_distance(fy,ft) = dc(ft);
	% 		end
	% 	end
	% 	ft_best = 0;
	% 	sum_best = 10000;
	% 	for ft=2:collection_frames-n_best
	% 		sum_total = 0;
	% 		for fy=2:(n_best)
	% 			sum_total = sum_total + matrix_distance(fy,ft+fy);
	% 		end
	% 		if sum_total < sum_best
	% 			sum_best = sum_total;
	% 			ft_best = ft;
	% 		end
	% 	end
	% 	% ft_best
	% 	% sum_best
	% 		% dc(1) = max(dc);
	% 		% % for i=1:length(ids)
	% 		% % 	dc(ids(i)) = max(dc);
	% 		% % end
	% 		% [a,idmin]= min(dc);
	% 		% ids = [ids idmin];
	% 		% [B,I] = sort(dc);
	% 	for fy=2:(n_best)
	% 		collecton_video_train_best(c,fy,:) = collecton_video_train(c,ft_best+fy,:);
	% 	end
	% 	% matrix_distance
	% end 

	% save([current_dir,'/collecton_video_train_best.mat'],'collecton_video_train_best');	

	Y = reshape(Y, [(n_best+1) feats]);
	ST  = LGSR(Y, collecton_video_train_best, type_distance, nexp);
	frame_selected = (n_best+1);

	Lc = zeros(nC2, 1);
	Qc = zeros(nC2, 1);
	for c=1:nC2
		X = reshape(collecton_video_train_best(c,:,:),[frame_selected feats]);
		STtmp = reshape(ST(c,:,:),[frame_selected frame_selected]);
		% STtmp
		xst = (X'*STtmp)';
		% Lc(c,1) = 0.5*norm( (Y - xst) ,'inf'); % max(svd((Y - xst ) ) ) ;
		% Qc(c,1) = norm( STtmp,'inf')   /  norm( (Y - xst ) ,'inf'); %max(svd(STtmp) ) /max(svd( (Y - xst )) );
		Lc(c,1) = 0.5*sum(sum(abs((Y - xst)).^nexp)).^(1/nexp); % max(svd((Y - xst ) ) ) ;
		Qc(c,1) = sum(sum(abs((STtmp)).^nexp)).^(1/nexp)  / sum(sum(abs((Y - xst)).^nexp)).^(1/nexp); %max(svd(STtmp) ) /max(svd( (Y - xst )) );
	
	end


	[a,idmin]= min(Lc);
	[amax,idmax]= max(Qc);
	[B,I] = sort(Lc);
	% id_c
	% Lc
	% % idmin
	% % idmax
	knn = 10;
	% if  Label_sub (idmin)  == 5
	% conf_matrix ( id_c, Label_sub (int8 (idmin) ) ) = conf_matrix (id_c, Label_sub (idmin) ) + 1;
	result(id,1) = Label_sub (idmin);
	result2(id,1) = Label_sub (idmax);
	% else
	% % 		B(1:5)
	% 	C=Label_sub (I); 
	% % C(1:5)

	% C=Label_sub (I); 
	% best_weight = 0;
	% best_label = 0;
	% best_n = 0;
	% for j = 1:4
	% 	id_knn = find(C(1:knn) == j);
	% 	if (length( id_knn) == best_n) 
	% 		best_n = length(id_knn );
	% 		best_label = j;
	% 		best_weight = B(id_knn);				
	% 	end
	% end
	% % best_label
	% p = best_label;%mode (C(1:knn));
	% p = mode (C(1:knn));
	% result(id,1) = Label_sub (p);
	% 	%TODO AGregar filtar por peso cuando hay dos modas ... para eso  B(1:knn) tiene los pesos e I el orden
	% 	conf_matrix ( id_c, p ) = conf_matrix (id_c, p) + 1;
	% end
	% conf_matrix2 ( id_c, Label_sub (int8 (idmax) ) ) = conf_matrix2 (id_c, Label_sub (idmax) ) + 1;

	% best_result1 = sum (diag(conf_matrix)) / sum(sum(conf_matrix));
	% best_result2 = sum (diag(conf_matrix2)) / sum(sum(conf_matrix2));
	% disp([num2str(i), ': ',num2str(best_result1), ' - ' num2str(best_result2)])

end

failsnew = zeros(40,1);
f=1;
failsnew2 = zeros(40,1);
f2=1;
for id=1:length(vector_test)
	i = vector_test(id);
	id_c = Label (int8(i));
	% id;lt(id)
	conf_matrix ( id_c, result(id) ) = conf_matrix (id_c, result(id) ) + 1;
	conf_matrix2 ( id_c, result2(id) ) = conf_matrix2 (id_c, result2(id) ) + 1;
	if id_c ~= result(id) 
		failsnew(f,1) = id;
		f=f+1;
	end
	if id_c ~= result2(id) 
		failsnew2(f2,1) = id;
		f2=f2+1;
	end
end

% if exist([current_dir,'/fails.mat'],'file')
%     load([current_dir,'/fails.mat']);
%     fails = [fails failsnew ];
% else
% 	fails = failsnew;
% end
% if exist([current_dir,'/fails2.mat'],'file')
%     load([current_dir,'/fails2.mat']);
%     fails2 = [fails2 failsnew2 ];
% else
% 	fails2 = failsnew2;
% end
% save([current_dir,'/fails.mat'],'fails');
% save([current_dir,'/fails2.mat'],'fails2');

best_result1 = sum (diag(conf_matrix)) / sum(sum(conf_matrix));
conf_matrix;
best_result2 = sum (diag(conf_matrix2)) / sum(sum(conf_matrix2));
disp([num2str(best_result1), ' - ' num2str(best_result2)])
conf_matrix2;
toc
% best_result = max (best_result1, best_result2);