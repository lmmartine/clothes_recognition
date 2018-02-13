function testresp = c1_online_classification(~,req,resp)
	

data_client = rossvcclient('/merge_data/get_sequence_data');
testreq = rosmessage(data_client);
testresp = call(data_client,testreq,'Timeout',10);
disp('Get sequence data ... OK');

video_features = c2_extract_features(testresp);
disp('Extract features  ... OK');

video_features = c3_local_features_llc(video_features);
disp('Apply LLC ... OK');

%crear collection
test_video = c4_features_integration(video_features, weight_feature);
ST  = LGSR(test_video, collecton_video, 'euclidean');
disp('Apply LGSR ... OK');

Y = reshape(Y, [collection_frames feats]);
Lc = zeros(nC2, 1);
% Qc = zeros(nC2, 1);
for c=1:nC2
	X = reshape(collecton_video(c,:,:),[collection_frames feats]);
	STtmp = reshape(ST(c,:,:),[collection_frames collection_frames]);
	% STtmp
	xst = (X'*STtmp)';
	Lc(c,1) = 0.5*norm( (Y - xst ) ,'inf'); % max(svd((Y - xst ) ) ) ;
	% Qc(c,1) = norm( STtmp,'inf')   /  norm( (Y - xst ) ,'inf'); %max(svd(STtmp) ) /max(svd( (Y - xst )) );
end

[a,idmin]= min(Lc);

disp('Classifcation  ... OK');

resp.ID = idmin;
