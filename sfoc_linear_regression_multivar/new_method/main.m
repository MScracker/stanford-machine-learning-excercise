clear all,close all,clc;
X = load('ex3x.dat');
y = load('ex3y.dat');
[m n] = size(X); 
X = [ones(m,1), X];
%initializing theta
init_theta = zeros(n + 1,1);

[cost, grad] = costfunction(init_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

options = optimset('GradObj','on','MaxIter',100);
costfunc = @(t)costfunction(t,X,y);
[theta J iterations] = fminunc(costfunc,init_theta,options);




fprintf('the minmium costfunction J is %f\n',J);
fprintf('and theta is \n');
fprintf('%f\n',theta);

