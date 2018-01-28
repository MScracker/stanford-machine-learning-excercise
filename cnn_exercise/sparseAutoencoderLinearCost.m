function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE -------------------- 
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
                                
%initializing
deltaW1 = zeros(size(W1));
deltaW2 = zeros(size(W2));
deltab1 = zeros(size(b1));
deltab2 = zeros(size(b2));
%forward
[n m] = size(data);
Z2 = W1*data + repmat(b1,1,m);
A2 = sigmoid(Z2);
Z3 = W2*A2 + repmat(b2,1,m);
A3 = Z3;

%auto-sparse
rho = (1/m).*sum(A2,2);
KL = sparsityParam.*log(sparsityParam./rho) + (1 - sparsityParam).*log((1-sparsityParam)./(1 - rho));
%backward
sparsityTerm = beta.*(- sparsityParam./rho + (1-sparsityParam)./(1-rho));
D3 = (A3 - data);
D2 = (W2'*D3 + repmat(sparsityTerm,1,m)).*sigmoidDeriv(Z2);
%derivative
%W2grad
W2grad = (1/m).*D3*A2' + lambda*W2;

%b2grad
b2grad = (1/m).*sum(D3,2);

%W1grad
W1grad = (1/m).*D2*data' + lambda*W1;

%b1grad
b1grad = (1/m).*sum(D2,2);

%J_sparse
cost = (0.5/m).*sum(sum((A3 - data).^2)) + (0.5*lambda).*sum(sum(W1.^2)) + (0.5*lambda).*sum(sum(W2.^2)) + beta*sum(KL);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function deriv = sigmoidDeriv(x)
	deriv = sigmoid(x).*(1 - sigmoid(x));
end
