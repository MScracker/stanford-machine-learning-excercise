close all,clear all, clc;
x = load('ex5Linx.dat');
y = load('ex5Liny.dat');
figure(1);
plot(x,y,'o','MarkerFaceColor','r','MarkerSize',5);

m = length(y);
X = [ones(m,1),x,x.^2,x.^3,x.^4,x.^5];
[m,n] = size(X);
theta = zeros(n,1);
plotStyle = {'b-','r-','k-'};
lambda = [0 1 10];

for i = 1: length(lambda);
	T = lambda(i)*eye(n);T(1) = 0;
	theta = (X'*X +T)\(X'*y);
	theta_norm = norm(theta)
	J = (0.5/m)*(X*theta - y)'*(X*theta - y)+(0.5/m)*lambda(i)*norm(theta([2:end]))^2
	hold on;
	a = linspace(-1,1,100)';
	b = [ones(length(a),1),a,a.^2,a.^3,a.^4,a.^5]*theta;
	plot(a,b,char(plotStyle(i)),'LineWidth',2);
end

