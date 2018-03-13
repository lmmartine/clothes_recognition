function  evaluate_lgsr_withclassfier_test(collecton_video, Label, vector_train, vector_test, SVMModel, type_distance)

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
n_class = 4;
conf_matrix = zeros(n_class,n_class);
conf_matrix2 = zeros(n_class,n_class);
	
collecton_video_sub = collecton_video(vector_train,:,:) ;
Label_sub = Label(vector_train,:,:) ;
[nC2, collection_frames, feats] = size(collecton_video_sub);
% [nC_sub, collection_frames_sub, feats_sub] = size(collecton_video_sub);

matrix_test = zeros (length(vector_test), nC2, collection_frames*collection_frames);


for id=1:length(vector_test)
	% i
	i = vector_test(id);
	% i
	id_c = Label (int8(i));
	id_c
	Y = collecton_video(i,:,:);

	ST  = LGSR(Y, collecton_video_sub, type_distance);
	[st1, st2, st3] = size(ST);

	Y = reshape(Y, [collection_frames feats]);
	Lc = zeros(nC2, 1);
	Qc = zeros(nC2, 1);
	for c=1:nC2
		% if Label_sub (int8 (c) ) ~= id_c
		% 	continue
		% end
		X = reshape(collecton_video_sub(c,:,:),[collection_frames feats]);
		STtmp = reshape(ST(c,:,:),[collection_frames collection_frames]);
		xst = (X'*STtmp)';
		Lc(c,1) = 0.5*norm( (Y - xst ) ,'inf'); % max(svd((Y - xst ) ) ) ;
		Qc(c,1) = norm( STtmp,'inf')   /  norm( (Y - xst ) ,'inf'); %max(svd(STtmp) ) /max(svd( (Y - xst )) );
	end
	% Lc
	[a,idmin]= min(Lc);
	[a,idmax]= max(Qc);
	[B,I] = sort(Lc);
	B(1:5)
	C=Label_sub (I); 
	C(1:5)

	ST2 = ST(I,:,:);
		% STvector = reshape(ST, [1 st1 st2*st3]);
	matrix_test = zeros (5 , st2*st3);

	for j = 1:5
		STvector = reshape(ST2(j,:,:), [1 st2*st3]);
		matrix_test(j ,: ) = STvector;
	end
	
	[label,score] = predict(SVMModel,matrix_test);
	label
	% [a,idmax]= max(score);
	% [a2,idmax2]= max(a);

	% disp([num2str(i), ' - ' num2str(id_c), '/ moda:',num2str( mode(label) ), ' - xscore' num2str(label (idmax(idmax2)) ),' score:',num2str( a2), ' length:' num2str(length(find(label == label (idmax(idmax2))  ))) , ' / ' num2str(length(find(label == id_c)))])
	

	% % idmin
	% % idmax

	% conf_matrix ( id_c, Label_sub (int8 (idmin) ) ) = conf_matrix (id_c, Label_sub (idmin) ) + 1;
	% conf_matrix2 ( id_c, Label_sub (int8 (idmax) ) ) = conf_matrix2 (id_c, Label_sub (idmax) ) + 1;

	% best_result1 = sum (diag(conf_matrix)) / sum(sum(conf_matrix));
	% best_result2 = sum (diag(conf_matrix2)) / sum(sum(conf_matrix2));
	% disp([num2str(i), ': ',num2str(best_result1), ' - ' num2str(best_result2)])

end

% best_result1 = sum (diag(conf_matrix)) / sum(sum(conf_matrix))
% conf_matrix
% best_result2 = sum (diag(conf_matrix2)) / sum(sum(conf_matrix2))
% conf_matrix2

% best_result = max (best_result1, best_result2);

	

	% for i = 1:st1
	% 	ids_subclass = Label_sub(i);
	% 	STvector = reshape(ST(ids_subclass,:,:), [1 1 collection_frames*collection_frames]);

	% 	if ids_subclass == id_c
	% 		matrix_train_true(id_c, traintrue ,: ) = STvector;
	% 		traintrue=traintrue + 1;
	% 	else
	% 		matrix_train_false(id_c, ids_subclass, trainfalse ,: ) = STvector;
	% 		trainfalse= trainfalse + 1 ;
	% 	end
	% end
	
[nlabel, n1, n2] = size(matrix_train_true);
matrix_train_true_2d = reshape(matrix_train_true(1,:,:), [n1 n2]);
matrix_train_true_2da = reshape(matrix_train_true(2,:,:), [n1 n2]);
matrix_train_true_2db = reshape(matrix_train_true(3,:,:), [n1 n2]);
matrix_train_true_2dc = reshape(matrix_train_true(4,:,:), [n1 n2]);
% matrix_train_false_2da = reshape(matrix_train_false(1, 1, :,:), [n1 n2]);
% matrix_train_false_2db = reshape(matrix_train_false(1, 3, :,:), [n1 n2]);
% matrix_train_false_2dc = reshape(matrix_train_false(1, 4, :,:), [n1 n2]);

% X=[ matrix_train_true_2d ;  matrix_train_true_2dc];
X=[matrix_train_true_2d; matrix_train_true_2da ; matrix_train_true_2db; matrix_train_true_2dc];
% X=[matrix_train_true_2d; matrix_train_false_2da ; matrix_train_false_2db; matrix_train_false_2dc];
% y = [ones(n1,1)*1 ; ones(n1,1)*4];
y = [ones(n1,1)*1  ; ones(n1,1)*2 ; ones(n1,1)*3 ; ones(n1,1)*4];

% SVMModel = fitcsvm(X,y);
SVMModel = fitcecoc(X,y);


% nlabel, matrix_train_true(1,:,:)


% matrix_train_false(1, 2,:,:)
% STvector = reshape(ST, [1 st1 st2*st3]);


% matrix_train_false(1, 3,:,:)
% matrix_train_false(1, 4,:,:)


% [label,score] = predict(SVMModel,newX);

% matrix_train_true = zeros (length(labels), nC_sub*nC_sub/4, collection_frames*collection_frames);
% matrix_train_false = zeros (length(labels), length(labels)-1, nC_sub*nC_sub/4, collection_frames*collection_frames);
