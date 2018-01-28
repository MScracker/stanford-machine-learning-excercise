clear all,close all,clc;

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  
hidden_layer_size = 25;   
num_labels = 10;

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
[m n] = size(X);

%% ================ Part 6: Initializing Pameters ================
fprintf('\nInitializing Neural Network Parameters ...\n')

W1 = randInitializeWeights(input_layer_size, hidden_layer_size);
W2 = randInitializeWeights(hidden_layer_size, num_labels);


%% =================== Part 8: Training NN ===================
fprintf('\nTraining Neural Network... \n')

%  You should also try different values of lambda
lambda = 1;
alpha = 0.1;
%load('ex4weights.mat');
%nn_params = [Theta1(:) ; Theta2(:)];
num_iters = 100;
S = zeros(num_iters,1);
for i = 1: num_iters
initial_nn_params = [W1(:) ; W2(:)];
[J grad] = bpnn_costfunction(initial_nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lambda);
Theta1_grad = reshape(grad(1:hidden_layer_size*(input_layer_size + 1)),hidden_layer_size,(input_layer_size + 1));
Theta2_grad = reshape(grad( (hidden_layer_size*(input_layer_size + 1) + 1):end ),num_labels,(hidden_layer_size + 1)); 
W1 = W1 - alpha*Theta1_grad;
W2 = W2 - alpha*Theta2_grad;
S(i) = J;    
end

plot(1:num_iters,S(1:num_iters),'r-');
pred = predict(W1, W2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


