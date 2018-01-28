clear all,close all;
x = load('ex3x.dat');
y = load('ex3y.dat');

m = length(y);
x = [ones(m,1),x];
theta = zeros(size(x(1,:)))';
alpha = 0.0000003; 
iterations = 100;
J = zeros(iterations,1);
%delta = 1;
%i = 2;
%grad = (1/m) .* x'*(x * theta - y);
%theta = theta - alpha .* grad;
%J(1) = (0.5/m)*(x*theta - y)'*(x*theta - y);
for i = 1:iterations 
	J(i) = (0.5/m).*(x * theta - y)'*(x * theta - y);
	grad = (1/m) .* x'*(x * theta - y);
	theta = theta - alpha .* grad;
	
	%delta = J(i-1)-J(i);
end

plot(1:iterations,J(1:iterations),'r-');
xlabel('numbers of iterations');
ylabel('Cost J');





