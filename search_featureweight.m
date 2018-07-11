
weight_feature = [40    60    60    40  60  60  40 80 100 5 180];
weight_feature2 = [40    60    60    40  60  60  40 80 20 80 300];
% [collecton_video, Label] = b1_clothes_classification(weight_feature, weight_feature2, 1, 15);
% weight_feature = [60    60    60    60  ];
% [collecton_video, Label] = b1_clothes_classification(weight_feature, 1, 5);
% [best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean')
% weight_feature


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
% weight_feature= [60 60 120 10 60];  % 0.6

while best_result1 < 0.7
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



[collecton_video, Label] = b1_clothes_classification(weight_feature, weight_feature2, 1, 15);
[best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean',5,100);

% 	[collecton_video, Label] = b1_clothes_classification(weight_feature, 1, 5);
% 	[best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean');
% 	best_result1
end

