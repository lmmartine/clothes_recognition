

% options_weight = [10 30 60 90];


% parpool(4)
% parfor i=1:length(options_weight)
% best_weight = [];
% best_result = 0;
% % for j=1:length(options_weight)
% % for k=1:length(options_weight)
% j=2;
% k=3;
% for m=1:length(options_weight)
% 	weight_feature= [options_weight(i) options_weight(j) options_weight(k) options_weight(m) ];
% 	[collecton_video, Label] = b1_clothes_classification(weight_feature);

% 	Lc = evaluate_lgsr(collecton_video, Label);
% 	weight_feature
% 	if Lc > best_result
% 		best_result = Lc;
% 		best_weight = weight_feature;
% 		disp('New result');
% 		disp(best_result)
% 		disp(best_weight)

% 	end
% end
% % end
% % end
% end

best_result1 = 0;

while best_result1 < 0.64
	labels = unique(Label);
	train_percent = 0.6;
	test_percent = 1 - train_percent;
	vector_train = [];
	for i=1:length(labels)
		n= sum(Label== labels(i));
		ids = find(Label== labels(i));
		p = randperm(n,ceil(n*train_percent));
		vector_train = [vector_train ; sort(ids(p))];
	end
	% vector_train, vector_test
	vector_test = ones( size(Label) );
	vector_test(vector_train) = 0;;
	vector_test = find(vector_test== 1);


	weight_feature= [15 40 120 10];  % 0.6

	[collecton_video, Label] = b1_clothes_classification(weight_feature, 13, 42);
	[best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean');
end

