close all,clear all, clc;
x = load('ex5Linx.dat');
y = load('ex5Liny.dat');
figure(1);
plot(x,y,'o','MarkerFaceColor','r','MarkerSize',5);

m = length(y);
X = [ones(m,1),x,x.^2,x.^3,x.^4,x.^5];
[m,n] = size(X);
theta = zeros(n,1);

lambda = [0 1 10];
alpha = [0.3];
plotstyle = {'b', 'r', 'g', 'k', 'y', 'm'};
num_iter = 100000;

for j = 1:length(lambda)
	J= zeros(num_iter,1);
	theta = zeros(n,1);
		for i = 1:num_iter
			h = X*theta;
			L = (1-alpha(1)*lambda(j)/m)*theta; L(1) = theta(1);
			%G = diag([1;(1-alpha*lambda(j)/m)*ones(n-1,1)])*theta;
			theta = L - alpha(1)/m.*X'*(h - y);
			J(i) = (0.5/m).*((h - y)'*(h-y)+(0.5/m).*lambda(j).*norm(theta([2:end])))^2;
		end
	%theta
	theta_norm = norm(theta)
	hold on;
	x_vals = (-1:0.05:1)';
	features = [ones(size(x_vals)), x_vals, x_vals.^2, x_vals.^3,x_vals.^4, x_vals.^5];
	plot(x_vals, features*theta, char(plotstyle(j)), 'LineWidth', 2)
	%a = (min(x)):0.01:(max(x));
	%b = theta(1)+a*theta(2)+a.^2*theta(3)+a.^3*theta(4)+a.^4*theta(5)+a.^5*theta(6);
	%plot(a,b,char(plotstyle(j)),'LineWidth',2);
	%figure(2);
	%plot(1:num_iter,J,char(plotstyle(j)),'LineWidth',2);
	%hold on;
end
%legend('Traning Data','\lambda=0','\lambda =1');

