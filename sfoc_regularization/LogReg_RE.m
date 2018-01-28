clear all,close all,clc;
x = load('ex5Logx.dat');
y = load('ex5Logy.dat');
pos = find(y==1);
neg = find(y==0);
figure(1)
plot(x(pos,1),x(pos,2),'+','LineWidth',2);
hold on;
plot(x(neg,1),x(neg,2),'o','MarkerFaceColor','y');


u = x(:,1);
v = x(:,2);
g = inline('1.0./(1.0 + exp(-z))');
num_iter = 20;
lambda = [1];
X = map_feature(u,v);
[m,n] = size(X);
plotStyle = {'ko-','ro-','bo-'};

for j = 1:length(lambda)
	theta = zeros(size(X(1,:)))';
	J = zeros(num_iter,1);
	for i = 1: num_iter
		z = X * theta;
		h = g(z);
		J(i) = (-1.0/m).*sum(y.*log(h)+(1-y).*log(1-h))+(0.5*lambda(j)/m).*norm(theta([2:end]))^2;
		%smart method
		 G = lambda(j)/m.*theta; G(1) = 0; 
		 L = lambda(j)/m.*eye(n);L(1) = 0;
		 grad = (1/m).*X'*(h -y) + G;
		 H = (1/m)*X'*diag(h)*diag(1-h)*X + L;
		 
		%grad = (1/m).*X'*(h -y) + diag([0;lambda(j)/m.*ones((length(theta)-1),1)])*theta;
		%H = (1/m)*X'*diag(h)*diag(1-h)*X + (lambda(j)/m.*diag([0;ones((length(theta)-1),1)]));
		theta = theta - H\grad;
		%norm error
		%J(i) = (-1.0/m).*sum(y.*log(h)+(1-y).*log(1-h))+(0.5*lambda(j)/m).*norm(theta);
	end
	theta_norm = norm(theta)
	%plot(0:num_iter-1,J,char(plotStyle(j)),'MarkerFaceColor', 'r', 'MarkerSize', 8);
	hold on;
	u = linspace(-1,1.5,200);
	v = linspace(-1,1.5,200);
	Z = zeros(length(u),length(v));
	for a = 1:length(u)
		for b = 1:length(v)
			Z(b,a) = map_feature(u(a),v(b))*theta; 
		end
	end
	contour(u,v,Z,[0,0],char(plotStyle(j)),'LineWidth',2);

end	
legend('y=1','y=0','\lambda = 1 Decision boundary');
hold off;





