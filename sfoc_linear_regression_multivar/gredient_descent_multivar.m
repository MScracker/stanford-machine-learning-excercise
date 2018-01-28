function [J,H,theta] = gredient_descent_multivar(x,y,alpha,num_iter)
ave = mean(x);
sigma = std(x);
x(:,1) = (x(:,1)-ave(1))/sigma(1);
x(:,2) = (x(:,2)-ave(2))/sigma(2);
m = length(y);
x = [ones(m,1),x];

theta = zeros(size(x(1,:)))';
Jot = zeros(num_iter,1);

for i = 1:num_iter
	Jot(i) = (0.5/m)*(x*theta - y)'*(x*theta - y);
	grad = (1/m)* x'*(x*theta - y);
	theta = theta - alpha*grad;
end
J = (0.5/m)*(x*theta - y)'*(x*theta - y);
H = [1,(1650-ave(1))/sigma(1),(3-ave(2))/sigma(2)]*theta;
plot(1:num_iter,Jot(1:num_iter),'r-');
