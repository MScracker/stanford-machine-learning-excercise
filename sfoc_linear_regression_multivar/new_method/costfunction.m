function [J grad] = costfunction(theta,X,y)
	m = length(y);
	J = (0.5/m)*(X*theta - y)'*(X*theta - y);
	grad = (1/m)*X'*(X*theta - y);	
end