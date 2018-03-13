function [emd_matix,euclidean_matrix]  = emd_matrix(videoP, videoQ)%, videoP_weight = 0, videoQ_weight = 0)

[n, P_frames, feat_P] = size(videoP);
[n, Q_frames, feat_Q] = size(videoQ);

emd_value = -1;

sigma = 0.2;

emd_matix =[];
euclidean_matrix=[];

if feat_P == feat_Q

	% if videoP_weight = 0
		videoP_weight = ones(P_frames,1)/feat_P;
		n_tableP = 4;
	% end

	% if videoQ_weight = 0
		videoQ_weight = ones(Q_frames,1)/feat_Q;
		n_tableQ = 4;
	% end

	emd_value = 0;
	for i=1:P_frames
	emd_line=[];
	euc_line = [];
	j=i;
	% for j=1:Q_frames
		fij = min(videoP_weight(i), videoQ_weight(j));
		v1=videoP(1,i,:);
		v1 = reshape(v1,[1 length(v1)]);
		v2=videoQ(1,j,:);
		v2 = reshape(v2,[1 length(v2)]);
		euclidean_dist = pdist([v1; v2]);
		fij_hat= fij*sum(euclidean_dist);

		emd_value = emd_value + fij_hat;

		% emd_line=[emd_line, exp(sum(euclidean_dist))/sigma]; % Agregar buscar el minimo y restarlo antes de sacar el exponencial
		euc_line = [euc_line, sum(euclidean_dist)];

	% end
	
	euclidean_matrix=[euclidean_matrix; euc_line];

	end

	emd_matix = exp((euclidean_matrix-min(min(euclidean_matrix)))/8) ; 
end

