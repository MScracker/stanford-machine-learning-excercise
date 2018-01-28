%function  [J,theta] = Logistic_Newton(X,y)
X = load('ex4x.dat');
y = load('ex4y.dat');
X_orig = X;
m = length(y);
X = [ones(m,1),X];
theta = zeros(size(X(1,:)))';
num_iter = 10;
g = inline('1.0 ./ (1.0 + exp(-z))');


J = zeros(num_iter,1);
for i = 1 : num_iter
	z = X * theta;
	h = g(z); 
	grad_J = (1/m) .* X' *( h - y); 
	H = (1/m).* X' * diag(h) * diag(1-h)*X;
	theta = theta - inv(H) * grad_J;
	J(i) = (1/m)*sum(-y.*log10(h) - (1-y).*log10(1-h));
end
theta
figure(1)
plot(1:num_iter,J,'r*-','MarkerFaceColor', 'r', 'MarkerSize', 3);


pos = find(y==1);
neg = find(y==0);
figure(2)
plot(X_orig(pos,1),X_orig(pos,2),'b+');
hold on;
plot(X_orig(neg,1),X_orig(neg,2),'ro');
xlabel('score 1');
ylabel('score 2');
line_x = [min(X_orig(:,1))-2,max(X_orig(:,1))+2];
%line_x = linspace(min(X_orig(:,1))-2,max(X_orig(:,1))+2,100);
line_y = (-1./theta(3))*(theta(1)+theta(2)*line_x);
plot(line_x,line_y,'y-');
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off;

J