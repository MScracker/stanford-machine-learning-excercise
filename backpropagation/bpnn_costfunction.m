function [J grad] = bpnn_costfunction(nn_params, input_layer_size,hidden_layer_size,...
									  num_labels, X, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
[m n]= size(X);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1),X];

S = zeros(m,1);
w1 = zeros(size(Theta1));
w2 = zeros(size(Theta2));

E = eye(num_labels);
for k = 1:num_labels
	index = find(y == k);
	Y(index,:) = repmat(E(k,:),length(index),1);
end

a1 = X;
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;
cost = -Y.*log(h)-(1-Y).*log(1-h);
Theta1_temp = Theta1; Theta1_temp(:,1) = 0;
Theta2_temp = Theta2; Theta2_temp(:,1) = 0;

reg = 0.5*lambda/m*( sum(Theta1_temp(:).^2) + sum(Theta2_temp(:).^2) );
J = (1/m).*sum(cost(:)) + reg;
%fprintf('J = %f\n',J);

% compute gradient
for i = 1:m
	a_1 = X(i,:)';
	z_2 = Theta1*a_1;
	a_2 = sigmoid(z_2);
	a_2 = [1;a_2];
	z_3 = Theta2*a_2;
	a_3 = sigmoid(z_3);
    p = a_3;
	delta_3 = a_3 - Y(i,:)';
	delta_2 = Theta2(:,2:end)'*delta_3.*sigmoidGradient(z_2);
	w2 = w2 + delta_3*a_2';
    w1 = w1 + delta_2*a_1';	
end
T1 = lambda/m*Theta1;T1(:,1) = 0; 
T2 = lambda/m*Theta2;T2(:,1) = 0;
Theta1_grad = (1/m)*w1 + T1;
Theta2_grad = (1/m)*w2 + T2;
grad = [Theta1_grad(:); Theta2_grad(:)];
end

