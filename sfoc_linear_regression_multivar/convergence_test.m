clear all,close all;
x = load('ex2x.dat');
y = load('ex2y.dat');

plot(x,y,'o');
xlabel('Age in years');
ylabel('Height in metres');

m = length(x);
x = [ones(m,1),x];
theta = zeros(2,1);
alpha = 0.07; 

J = zeros(1000,1);
delta = 1;
i = 2;
grad = (1/m) .* x'*(x * theta - y);
theta = theta - alpha .* grad;
J(1) = (0.5/m)*(x*theta - y)'*(x*theta - y);
while (delta >0.0001)
	grad = (1/m) .* x'*(x * theta - y);
	theta = theta - alpha .* grad;
	J(i) = (0.5/m)*(x*theta - y)'*(x*theta - y);
	delta = J(i-1)-J(i)
	%delta
	i = i + 1;
end
theta

plot(1:i-1,J(1:i-1),'-');
xlabel('numbers of iterations');
ylabel('Cost J');





