function [J,Predict,theta] = gredient_descent_normal_equation(x,y)
m = length(y);
x = [ones(m,1),x];
theta = inv(x'*x)*x'*y;
J = (0.5/m)*(x*theta - y)'*(x*theta - y);
Predict = [1 1650 3]*theta;