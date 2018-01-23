function Lc = evaluate_lgsr(collecton_video, Label, i)

% nCollection=17;
% nC=17;
addpath('./LLC');
addpath('./drtoolbox');

%% read code book 
current_dir='/home/lmartinez/bags';
codebook_dir = [current_dir,'/Features/'];
kofkmeans = 256;
pooling_opt = 'sum';
knn = 5;
load([codebook_dir,'code_book',num2str(kofkmeans),'.mat']);


collecton_video_llc=collecton_video;
[nC, collection_frames, feats] = size(collecton_video_llc);
for i=1:nC
	for j=1:collection_frames
		X = reshape(collecton_video_llc(i,j,:), [1 feats]);
		collecton_video_llc(i,j,:) = LLC(X, [1 10 30], 3, [0.3 0.3 0.3], [0.3 0.3 0.3], 'Matlab'); %LLC_pooling( X, code_book.bsp, code_book.bsp_weights, knn, pooling_opt );
	end
end


[nC, collection_frames, feats] = size(collecton_video);
n_class = 4;
conf_matrix = zeros(n_class,n_class);
conf_matrix2 = zeros(n_class,n_class);

% r1 = randi(nC,10,1);

% for id=1:length(r1)
for i=1:nC
	% i = r1(id);
	id_c = Label (int8(i));
	Y = collecton_video(i,:,:);
	collecton_video_sub =[];
	Label_sub =[];

	if i > 1
		collecton_video_sub = [collecton_video(1:i-1,:,:) ; collecton_video(i+1:nC,:,:) ];
		Label_sub = [Label(1:i-1,:,:) ; Label(i+1:nC,:,:) ];
	else
		collecton_video_sub = collecton_video(i+1:nC,:,:) ;
		Label_sub = Label(i+1:nC,:,:) ;
	end


	[nC2, collection_frames, feats] = size(collecton_video_sub);
	ST  = LGSR(Y, collecton_video_sub);

	Y = reshape(Y, [collection_frames feats]);
	Lc = zeros(nC2, 1);
	Qc = zeros(nC2, 1);
	for c=1:nC2
		X = reshape(collecton_video_sub(c,:,:),[collection_frames feats]);
		STtmp = reshape(ST(c,:,:),[collection_frames collection_frames]);
		xst = (X'*STtmp)';
		Lc(c,1) = 0.5* norm( (Y - xst ) ,'fro');
		Qc(c,1) = norm( STtmp,'fro')   /  norm( (Y - xst ) ,'fro');
	end

	[a,idmin]= min(Lc);
	[a,idmax]= max(Qc);
	idmin
	idmax

	conf_matrix ( id_c, Label_sub (int8 (idmin) ) ) = conf_matrix (id_c, Label_sub (idmin) ) + 1;
	conf_matrix2 ( id_c, Label_sub (int8 (idmax) ) ) = conf_matrix2 (id_c, Label_sub (idmax) ) + 1;
end

sum (diag(conf_matrix)) / sum(sum(conf_matrix))
conf_matrix
sum (diag(conf_matrix2)) / sum(sum(conf_matrix2))
conf_matrix2
