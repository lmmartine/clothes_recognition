function emd_value = emd(videoP, videoQ)%, videoP_weight = 0, videoQ_weight = 0)

[n, P_frames, feat_P] = size(videoP);
[n, Q_frames, feat_Q] = size(videoQ);

emd_value = -1;
method = 3;

if feat_P == feat_Q

	% if videoP_weight = 0
		videoP_weight = ones(P_frames,1)/P_frames;
		n_tableP = 4;
	% end

	% if videoQ_weight = 0
		videoQ_weight = ones(Q_frames,1)/Q_frames;
		n_tableQ = 4;
	% end

	emd_value = 0;

	if method ==1 
		for i=1:P_frames
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
		% end
		end
	elseif method ==2
		
		for i=1:n_tableP
		for j=1:n_tableQ
			fij = min(videoP_weight(i), videoQ_weight(j));
			v1=videoP(1,i,:);
			v1 = reshape(v1,[1 length(v1)]);
			v2=videoQ(1,j,:);
			v2 = reshape(v2,[1 length(v2)]);
			euclidean_dist = pdist([v1; v2]);
			fij_hat= fij*sum(euclidean_dist);

			emd_value = emd_value + fij_hat;
		end
		end

		for i=n_tableP+1:P_frames
		for j=n_tableQ+1:Q_frames
			fij = min(videoP_weight(i), videoQ_weight(j));
			v1=videoP(1,i,:);
			v1 = reshape(v1,[1 length(v1)]);
			v2=videoQ(1,j,:);
			v2 = reshape(v2,[1 length(v2)]);
			euclidean_dist = pdist([v1; v2]);
			fij_hat= fij*sum(euclidean_dist);

			emd_value = emd_value + fij_hat;
		end
		end
	elseif method ==3
		
		for i=1:P_frames
		for j=1:Q_frames
			fij = min(videoP_weight(i), videoQ_weight(j));
			v1=videoP(1,i,:);
			v1 = reshape(v1,[1 length(v1)]);
			v2=videoQ(1,j,:);
			v2 = reshape(v2,[1 length(v2)]);
			euclidean_dist = pdist([v1; v2]);
			fij_hat= fij*sum(euclidean_dist);

			emd_value = emd_value + fij_hat;
		end
		end

	end

end

