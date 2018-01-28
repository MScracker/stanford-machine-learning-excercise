close all,clear all,clc;
% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
fprintf('Loading data...\n');
load ('ex5data1.mat');

m = size(X, 1);

%Feature Mapping for Polynomial Regression 
p = 8;
% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones


%  Train linear regression with lambda = 0
fprintf('Training Linear Regression model...\n');
lambda = 3;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 2, 'LineWidth', 1);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));


% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones
error_test = linearRegCostFunction(X_poly_test,ytest,theta,0)
fprintf('error_test is %f when lambda is %f\n',error_test,lambda);
