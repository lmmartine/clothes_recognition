function [ST]  = lgsr(videoP, collection_videos,type_distance)%, videoP_weight = 0, videoQ_weight = 0)
% type_distance
[P_frames, feat_P] = size(videoP)
[nC, collection_frames, feats] = size(collection_videos)

% if not feat_P == feats
% 	return
% end
show_numer = false;
numer = 15;

it=0;
convergence = false;
it_max=1000; % 84,27
sigma= 1/40;% 1/2..1/4 ... 1/8 .. 1/40
epsilon = 0.001; % 001
betha =1/1000000; %1000000,27
gama = 1/16; % o 1/8 o 1/4 o 1/16


Y = videoP;%reshape(videoP, [P_frames feat_P]);

%Dc between Y and collection videos
D_norm = true;
norm_dc = true;
dc = zeros(nC, 1);
for c=1:nC
	dc(c) = EMD(videoP , collection_videos(c,:,:),type_distance);
end 
if norm_dc
	dc = dc / sum(dc);
end

Dc = zeros(nC, P_frames, P_frames);
for c=1:nC
	for i=1:P_frames
	v1=videoP(i,:);
	v1 = reshape(v1,[1 length(v1)]);
		for j=1:P_frames
			
			v2=collection_videos(c,j,:);
			v2 = reshape(v2,[1 length(v2)]);

		            v1 = real (v1);
            v2 = real (v2);
            
			Dc(c,i,j) = exp((dc(c)-min(dc))/sigma)*pdist([v1; v2], type_distance) ;

		end
	end
	if c==numer && show_numer
		dc(c)
		min(dc)
		exp((dc(c)-min(dc))/sigma)
		Dc(c,:,:)
	end
	if D_norm
		Dc(c,:,:)= Dc(c,:,:)/sum(sum(Dc(c,:,:)));
	end
end



ST = zeros(nC, P_frames, P_frames);
ST_next = ST;
A=[];
ready = [];

while it < it_max && convergence == false
	Lc = zeros(nC, 1);
	nocandidate = [];
	for i=1:nC
		X = reshape(collection_videos(i,:,:),[collection_frames feats]);
		STtmp = reshape(ST(i,:,:),[P_frames P_frames]);
		xst = (X'*STtmp)';
		Lc(i,1) =norm( (X)*(xst - Y)' ,'inf'); % max(svd( (X)*(xst - Y)') 
		% size (Lc)
		if min(min(ST(i,:,:))) ~= 0 || max(max(ST(i,:,:))) ~= 0 
			% Lc(i,1) = -1;
			nocandidate(length(nocandidate)+1) = i;
		end
	end
	Lc(nocandidate,1) = min(Lc);

	[maxval, idmax_Lc] = max(Lc);
	if  length(find(A == idmax_Lc))==0 && Lc(idmax_Lc,1) > gama*min(dc)
		A(length(A)+1) = idmax_Lc;
	% else
	% 	idmax_Lc
	end

	for cinA = 1:length (A)
		valA = A(cinA);
		% valA
		if length(find(ready == valA))>0
			continue
		end
		X = reshape(collection_videos(valA,:,:),[collection_frames feats]);
		Dctmp = Dc(valA,:, :);
		STtmp = ST(valA,:,:);
		Dctmp = reshape(Dctmp,[P_frames P_frames]);
		STtmp = reshape(STtmp,[P_frames P_frames]);

		xst = (X'*STtmp)';
		Lc_tmp =  X*(xst-Y)';
		if valA==numer && show_numer
			Lc_tmp
		end

		DS = 0;
		if min(min(STtmp)) ~= 0 || max(max(STtmp)) ~= 0 
			DS = ( (Dctmp.*Dctmp.*STtmp)/norm( Dctmp.*STtmp ,'inf') ) ; %max(svd( Dctmp.*STtmp)) ) 
			if valA==numer && show_numer
				Dctmp
				norm( Dctmp.*STtmp ,'inf')
				DS
			end
		end
		
		GS = Lc_tmp + gama * DS  ;
		GS = reshape(GS,[1 P_frames P_frames]);

		ST_next(valA,:,:) = ST(valA,:,:) - betha * GS;
		% ST_next(valA)
		% if ST_next(valA) == 0
		% 	A(cinA) = [];
		% end

		sttmp = ST_next(valA,:,:) - ST(valA,:,:);
		sttmp = reshape(sttmp,[P_frames P_frames]);
		if   norm(sttmp,'inf') < epsilon %max(svd(sttmp) ) <epsilon
			ready(length(ready)+1) = valA;
			% convergence = true
			% va/lA
		end

	end

	it=it+1;

	ST=ST_next;
	% if show_numer
	% ST_next(3,:,:)
	% end
	
end


