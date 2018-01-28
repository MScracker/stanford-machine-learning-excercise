function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
%for imageNum = 1:numImages
%  for featureNum = 1:numFilters
%	for i = 1 : floor(convolvedDim / poolDim)
%		for j = 1: floor(convolvedDim / poolDim)
%			row_start = poolDim*(i-1)+1; row_end = poolDim*i;
%			col_start = poolDim*(j-1)+1; col_end = poolDim*j;
%			meanFeatrues = convolvedFeatures(row_start:row_end,col_start:col_end,featureNum,imageNum);
%			pooledFeatures( i, j, featureNum, imageNum) = mean(meanFeatrues(:));
%		end
%	end
% end
%end

for imageNum = 1:numImages
  for featureNum = 1:numFilters
	crossConv = conv2(convolvedFeatures(:,:,featureNum,imageNum),ones(poolDim),'valid');
	pooledFeatures(:,:,featureNum,imageNum) = crossConv(1:poolDim:end,1:poolDim:end)./(poolDim*poolDim);
  end
end

end

