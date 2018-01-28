function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    
    cost = 0;
    grad = zeros(numFeatures, visibleSize);
    num_samples = size(patches,2);
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE -------------------- 
    lambda = 1e-2;
    m = size( patches , 2);
    w = weightMatrix;
    x = patches;
    error = w'*w*x - x;
    cost = 1/m * sum( error(:).^2 ) + lambda * sum( sum ( sqrt( (w*x).^2 + epsilon ) ) );
    grad = 1/m * ( 2*w*(w'*w*x-x)*x' + 2*(w*x)*(w'*w*x-x)' ) + ...
          lambda * ( (w*x).^2 + epsilon).^(-0.5).*(w*x) * x';
    grad = grad(:);
end