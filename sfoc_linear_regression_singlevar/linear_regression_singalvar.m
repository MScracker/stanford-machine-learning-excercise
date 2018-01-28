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
iterations = 1500;

for i = 1 : 1500 
	grad = (1/m) .* x'*(x * theta - y);
	theta = theta - alpha .* grad; 
end

theta


H = x * theta;
hold on;
plot(x(:,2),H);
legend('Training Data','Linear Regression');
predit1 = [1 3.5;1 7]* theta
text([3.5 7],predit1,'*','color','g');
hold off;

theta0 = linspace(-3, 3, 100);
theta1 = linspace(-1, 1, 100);

J = zeros(length(theta0),length(theta1));
for i = 1:length(theta0)
	for j = 1:length(theta1)
		t = [theta0(i); theta1(j)];
		J(i,j) = (0.5/m)*(x*t-y)'*(x*t-y);
	end
end
J = J';
figure
mesh(theta0,theta1,J);
xlabel('\theta_0');
ylabel('\theta_1');

figure
contour(theta0,theta1,J,logspace(-2, 2, 15));
xlabel('\theta_0');
ylabel('\theta_1');






 