function [matrix_train_true, matrix_train_false] = evaluate_lgsr_withclassifier(collecton_video, Label, vector_train, vector_test, type_distance)

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


labels = unique(Label);


[nC, collection_frames, feats] = size(collecton_video);
n_class = 5;
conf_matrix = zeros(n_class,n_class);
conf_matrix2 = zeros(n_class,n_class);
	
collecton_video_sub = collecton_video(vector_train,:,:) ;
Label_sub = Label(vector_train,:,:) ;
[nC_sub, collection_frames_sub, feats_sub] = size(collecton_video_sub);

matrix_train_true = zeros (length(labels), nC_sub*nC_sub/(n_class*n_class), collection_frames*feats_sub);
matrix_train_false = zeros (length(labels), length(labels), nC_sub*nC_sub/(n_class*n_class), collection_frames*feats_sub);
	traintrue = 1;
	trainfalse = 1;
	lastc=1;
	lastl=1;
for id=1:length(vector_train)


	i = vector_train(id);
	id_c = Label (int8(i));
	Y = collecton_video(i,:,:);
	Y = reshape(Y, [collection_frames_sub feats_sub]);
	if lastc ~= id_c
		traintrue = 1;
		trainfalse = 1;
		lastl=1;
	end

	lastc = id_c;

	[nC2, collection_frames, feats] = size(collecton_video_sub);
	ST  = LGSR(Y, collecton_video_sub, type_distance);
	[st1, st2, st3] = size(ST);
	ST;
	for i = 1:st1
		ids_subclass = Label_sub(i);
		if lastl ~= ids_subclass
			trainfalse = 1;
		end
		lastl=ids_subclass;

		xst = collecton_video_sub(i,:,:);
		xst = reshape(xst, [collection_frames_sub feats_sub]);
		STvector = reshape(ST(ids_subclass,:,:), [collection_frames collection_frames]);

		trainvector = xst'*STvector;
		[tv1, tv2] = size(trainvector);
		trainvector = reshape(trainvector, [1 1 tv1*tv2]);

		if ids_subclass == id_c

			matrix_train_true(id_c, traintrue ,: ) = trainvector;
			traintrue=traintrue + 1;
		else
			matrix_train_false(id_c, ids_subclass, trainfalse ,: ) = trainvector;
			trainfalse= trainfalse + 1 ;
		end
	end

	
end
