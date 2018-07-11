

options_len_images = [15 ]; % 30 40
weight_feature = [40    60    60    40  60  60  40 80 100 5 180];
weight_feature2 = [40    60    60    40  60  60  40 80 20 80 300];
% weight_feature= [15 40 120 10];
max_img = 45;
exp_results1 = zeros (length(options_len_images), 100) ;
exp_results2 = exp_results1;
exp_resultsX = exp_results1;
% exp_resultsX2 = exp_results1;
% parpool(4)
for i=1:length(options_len_images)
	len_images = options_len_images(i);
	imginit = 25;
	imgend = imginit+len_images-1;
	id_exp = 1;
% best_weight = [];
% best_result = 0;

% j=2;
% k=3;
	while imgend  <= max_img %&& id_exp <3
		imginit
		imgend
		% [collecton_video, Label] = b1_clothes_classification(weight_feature, imginit, imgend);
		% [best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean');
		[collecton_video, Label] = b1_clothes_classification(weight_feature, weight_feature2, imginit, imgend);
		[best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean',5,100);

		exp_results1 (i, id_exp) = best_result1;
		exp_results2 (i, id_exp) = best_result2;
		exp_resultsX (i, id_exp) = imgend;
		imginit = imginit + 3;
		imgend = imginit + len_images -1;
		id_exp = id_exp + 1;
	% Lc = evaluate_lgsr(collecton_video, Label);
	% weight_feature
	% if Lc > best_result
	% 	best_result = Lc;
	% 	best_weight = weight_feature;
	% 	disp('New result');
	% 	disp(best_result)
	% 	disp(best_weight)

	% end
    end
end



		% [collecton_video, Label] = b1_clothes_classification([40 40 80 10], imginit, imgend);
		% [best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean');

		% 		[collecton_video, Label] = b1_clothes_classification([40 40 60 10], imginit, imgend);
		% [best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean');