function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    
    cost = 0;
    grad = zeros(numFeatures, visibleSize);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------     
	[n m] = size(patches);
	%fobj = weightMatrix'*weightMatrix*patches - patches;
	%cost = (1/m)*sum(fobj(:).^2);
	featrues = weightMatrix*patches ;
	cost = (1/m)*sum(sqrt(featrues(:).^2 + epsilon));
	%grad = (2/m)*weightMatrix*(weightMatrix'*weightMatrix*patches - patches)*patches' + (2/m)*weightMatrix*patches*(weightMatrix'*weightMatrix*patches - patches)';
	grad = (1/m)*weightMatrix*patches./sqrt(featrues.^2 + epsilon)*patches';
	grad = grad(:);
end

