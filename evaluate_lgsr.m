function [best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, type_distance)

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

% LLC al conjunto de caracteristicas del video, reduciendolo a 10, es el mejor resultado encontrado
% collecton_video_llc=[];
% [nC, collection_frames, feats] = size(collecton_video);
% for i=1:nC
% 	% i
% 	X = reshape(collecton_video(i,:,:), [collection_frames feats]);
% 	[mappedXllc, mapping] = compute_mapping(X, 'Autoencoder',500);
% 	% size(X)
% 	% [mappedXllc, mapping] = compute_mapping(X, 'LLC',400);
% 	% mappedXllc= [mappedXpca; mappedXllc] ;
% 	% size(mappedXllc)
% 	mappedXllc = reshape(mappedXllc, [1 collection_frames 500]);
% 	collecton_video_llc(i,:,:) = mappedXllc;
% end

% collecton_video = collecton_video_llc;




[nC, collection_frames, feats] = size(collecton_video);
n_class = length(unique(Label));
conf_matrix = zeros(n_class,n_class);
conf_matrix2 = zeros(n_class,n_class);
	
collecton_video_sub = collecton_video(vector_train,:,:) ;
Label_sub = Label(vector_train,:,:) ;
	

% r1 =[4 6 12 15  19 23 26 32  39 45 48 52  58 62  67 70]; %randi(nC,30,1);

% for id=1:length(r1)
for id=1:length(vector_test)
	% i
	i = vector_test(id);
	id_c = Label (int8(i));
	Y = collecton_video(i,:,:);
	% collecton_video_sub =[];
	% Label_sub =[];

	% if i > 1
	% 	collecton_video_sub = [collecton_video(1:i-1,:,:) ; collecton_video(i+1:nC,:,:) ];
	% 	Label_sub = [Label(1:i-1,:,:) ; Label(i+1:nC,:,:) ];
	% else
	% 	collecton_video_sub = collecton_video(i+1:nC,:,:) ;
	% 	Label_sub = Label(i+1:nC,:,:) ;
	% end



    [nC2, collection_frames, feats] = size(collecton_video_sub);
	ST  = LGSR(Y, collecton_video_sub, type_distance);

	Y = reshape(Y, [collection_frames feats]);
	Lc = zeros(nC2, 1);
	Qc = zeros(nC2, 1);
	for c=1:nC2
		X = reshape(collecton_video_sub(c,:,:),[collection_frames feats]);
		STtmp = reshape(ST(c,:,:),[collection_frames collection_frames]);
		% STtmp
		xst = (X'*STtmp)';
		Lc(c,1) = 0.5*norm( (Y - xst ) ,'inf'); % max(svd((Y - xst ) ) ) ;
		Qc(c,1) = norm( STtmp,'inf')   /  norm( (Y - xst ) ,'inf'); %max(svd(STtmp) ) /max(svd( (Y - xst )) );
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
		conf_matrix ( id_c, Label_sub (int8 (idmin) ) ) = conf_matrix (id_c, Label_sub (idmin) ) + 1;
	% else
	% % 		B(1:5)
	% 	C=Label_sub (I); 
	% % C(1:5)

	% % 	C=Label_sub (I); 
	% % 	best_weight = 0;
	% % 	best_label = 0;
	% % 	best_n = 0;
	% % 	for j = 1:4
	% % 		id_knn = find(C(1:knn) == j);
	% % 		if length( id_knn) > best_n || ( (length( id_knn) == best_n) && (B(id_knn) < best_weight) )
	% % 			best_n = length(id_knn );
	% % 			best_label = j;
	% % 			best_weight = B(id_knn);				
	% % 		end
	% % 	end
	% % 	best_label
	% 	% p = best_label;%mode (C(1:knn));
	% 	p = mode (C(1:knn));
	% 	%TODO AGregar filtar por peso cuando hay dos modas ... para eso  B(1:knn) tiene los pesos e I el orden
	% 	conf_matrix ( id_c, p ) = conf_matrix (id_c, p) + 1;
	% end
	conf_matrix2 ( id_c, Label_sub (int8 (idmax) ) ) = conf_matrix2 (id_c, Label_sub (idmax) ) + 1;

	% best_result1 = sum (diag(conf_matrix)) / sum(sum(conf_matrix));
	% best_result2 = sum (diag(conf_matrix2)) / sum(sum(conf_matrix2));
	% disp([num2str(i), ': ',num2str(best_result1), ' - ' num2str(best_result2)])

end

best_result1 = sum (diag(conf_matrix)) / sum(sum(conf_matrix));
conf_matrix;
best_result2 = sum (diag(conf_matrix2)) / sum(sum(conf_matrix2));
disp([num2str(best_result1), ' - ' num2str(best_result2)])
% conf_matrix2

% best_result = max (best_result1, best_result2);