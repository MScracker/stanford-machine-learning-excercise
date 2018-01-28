function [J,theta,delta] = iter_check(x,y,iterations)
m = length(x);
x = [ones(m,1),x];
theta = zeros(size(x(1,:)))';
alpha = 0.07; 
J = zeros(iterations,1);
delta = 1;
%i = 2;
grad = (1/m) .* x'*(x * theta - y);
theta = theta - alpha .* grad;
J(1) = (0.5/m)*(x*theta - y)'*(x*theta - y);
for i = 2:iterations 
	grad = (1/m) .* x'*(x * theta - y);
	theta = theta - alpha .* grad;
	J(i) = (0.5/m)*(x*theta - y)'*(x*theta - y);
	delta = J(i-1)-J(i);
end
plot(1:i,J(1:i),'r-');
xlabel('numbers of iterations');
ylabel('Cost J');