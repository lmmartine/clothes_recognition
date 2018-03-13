

options_len_images = [20 ]; % 30 40
weight_feature= [15 40 120 10];
max_img = 55;
exp_results1 = zeros (length(options_len_images), 100) ;
exp_results2 = exp_results1;
% parpool(4)
for i=1:length(options_len_images)
	len_images = options_len_images(i);
	imginit = 1;
	imgend = 1+len_images-1;
	id_exp = 1;
% best_weight = [];
% best_result = 0;

% j=2;
% k=3;
	while imgend  < max_img %&& id_exp <3
		% imginit
		% imgend
		[collecton_video, Label] = b1_clothes_classification(weight_feature, imginit, imgend);
		[best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean');
		exp_results1 (i, id_exp) = best_result1;
		exp_results2 (i, id_exp) = best_result2;
		imginit = imginit + 1;
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

				[collecton_video, Label] = b1_clothes_classification([40 40 60 10], imginit, imgend);
		[best_result1, best_result2] = evaluate_lgsr(collecton_video, Label, vector_train, vector_test, 'euclidean');