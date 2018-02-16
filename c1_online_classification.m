function testresp = c1_online_classification(~,req,resp)
	
% global weight_feature
global collecton_video
global Label
weight_feature= [40 40 60 10]; 

data_client = rossvcclient('/merge_data/get_sequence_data');
testreq = rosmessage(data_client);
testresp = call(data_client,testreq,'Timeout',10);
disp('Get sequence data ... OK');

tic
video_features = c2_extract_features(testresp);
disp( [ 'Extract features  ... OK (' int2str(toc) 'sec)']);


tic
video_features = c3_local_features_llc(video_features);
disp( [ 'Apply LLC ... OK (' int2str(toc) 'sec)']);


%crear collection
test_video = c4_features_integration(video_features, weight_feature);
tic
ST  = LGSR(test_video, collecton_video, 'euclidean');
disp( [ 'Apply LGSR ... OK (' int2str(toc) 'sec)']);


tic
[nC, collection_frames, feats] = size(collecton_video);
Lc = zeros(nC, 1);
% Qc = zeros(nC, 1);
for c=1:nC
	X = reshape(collecton_video(c,:,:),[collection_frames feats]);
	STtmp = reshape(ST(c,:,:),[collection_frames collection_frames]);
	% STtmp
	xst = (X'*STtmp)';
	Lc(c,1) = 0.5*norm( (test_video - xst ) ,'inf'); % max(svd((Y - xst ) ) ) ;
	% Qc(c,1) = norm( STtmp,'inf')   /  norm( (Y - xst ) ,'inf'); %max(svd(STtmp) ) /max(svd( (Y - xst )) );
end

[a,idmin]= min(Lc);

disp(['Classifcation  ... OK (' int2str(toc) 'sec)']);
category = {'pant','shirt','sweater','tshirt'};
disp(['Class : ' int2str(Label(idmin)) category{Label(idmin)} ]);
resp.ID = Label(idmin);


data_client = rossvcclient('/merge_data/set_result');
req = rosmessage(data_client);
req.ID = Label(idmin);
testresp = call(data_client,req,'Timeout',10);
