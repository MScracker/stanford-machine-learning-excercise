function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%




%theta_stack_1 = [stack{1}.w(:); stack{1}.b(:)];
%theta_stack_2 = [stack{2}.w(:); stack{2}.b(:)];
%[costX1, stackgrad1] = sparseAutoencoderCost(theta_stack_1, inputSize, netconfig.layersizes{1}, 0, netconfig.sparsityParam, netconfig.beta, data);
%[featureStack1] = feedForwardAutoencoder(theta_stack_1, netconfig.layersizes{1}, inputSize, data);

%[costX2, stackgrad2] = sparseAutoencoderCost(theta_stack_2, netconfig.layersizes{1},netconfig.layersizes{2}, 0, netconfig.sparsityParam, netconfig.beta, featureStack1);
%[featureStack2] = feedForwardAutoencoder(theta_stack_2, netconfig.layersizes{2}, netconfig.layersizes{1}, featureStack1);

%[cost, softmaxThetaGrad] = softmaxCost(softmaxTheta, numClasses, netconfig.layersizes{2}, lambda, featureStack2, labels);

%stackgrad{1}.w = reshape(stackgrad1(1:netconfig.layersizes{1}*inputSize),netconfig.layersizes{1},inputSize);
%stackgrad{1}.b = reshape(stackgrad1(netconfig.layersizes{1}*inputSize+1 : end),netconfig.layersizes{1},1);

%stackgrad{2}.w = reshape(stackgrad1(1:netconfig.layersizes{2}*netconfig.layersizes{1}),netconfig.layersizes{2},netconfig.layersizes{1});
%stackgrad{2}.b = reshape(stackgrad1(netconfig.layersizes{2}*netconfig.layersizes{1}+1 : end),netconfig.layersizes{2},1);



%forward propagation
depth = numel(stack);
Z = cell(depth+1,1);
A = cell(depth+1,1);
D = cell(depth+1,1);
A{1} = data;
for layer = 1 : depth 
	Z{layer + 1}= stack{layer}.w * A{layer} + repmat(stack{layer}.b,1,numCases);
	A{layer + 1} = sigmoid(Z{layer + 1});
end 

%softmax
M = softmaxTheta*A{depth+1};
M = bsxfun(@minus,M,max(M,[],1));
M = exp(M);
p = bsxfun(@rdivide,M,sum(M));
cost = (-1.0/numCases).*groundTruth(:)'*log(p(:)) + lambda/2.0 *sum(softmaxTheta(:).^2);
softmaxThetaGrad = (-1.0/numCases).*(groundTruth - p)*A{depth +1}' + lambda*softmaxTheta;

%finetuning
D{depth + 1} = -softmaxTheta'*(groundTruth - p).*sigmoidDiff(Z{depth+1});
for layer = depth :-1: 2
	D{layer} = stack{layer}.w'*D{layer+1} .* sigmoidDiff(Z{layer});

end
for layer = depth :-1: 1
	stackgrad{layer}.w = (1/numCases)*D{layer+1}*A{layer}';
	stackgrad{layer}.b = (1/numCases)*sum(D{layer+1},2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function diff = sigmoidDiff(x)
	diff = sigmoid(x).*(1 - sigmoid(x));
end