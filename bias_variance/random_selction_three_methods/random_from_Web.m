clear all,close all,clc;
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

%%Optional (ungraded) exercise: Plotting learning curves with randomly selected examples
 m=size(X_poly,1);
 X_poly_y=[X_poly,y];
 X_poly_val_y=[X_poly_val,yval];
 lambda= 3;
 error_train = zeros(m, 1);
 error_val   = zeros(m, 1);
 for i=1:m
    error_train_sum=0;
    error_val_sum  =0;
    for k=1:50                           %50次迭代

         sel = randperm(m);
		 sel = sel(1:i);
		 rand_X_poly_y=X_poly_y(sel,:);
         rand_X_poly_val_y=X_poly_val_y(sel,:);
         X=rand_X_poly_y(:,1:end-1);
         y=rand_X_poly_y(:,end);
         Xval=rand_X_poly_val_y(:,1:end-1);
         yval=rand_X_poly_val_y(:,end);
         theta=trainLinearReg(X,y,lambda);
         [error_train_val,grad]=linearRegCostFunction(X, y, theta, 0);
         [error_val_val,  grad]=linearRegCostFunction(Xval, yval, theta, 0);
         error_train_sum=error_train_sum+error_train_val;
         error_val_sum=error_val_sum+error_val_val;
    end
    error_train(i)=error_train_sum/50;
    error_val(i)=error_val_sum/50;
	%error_val(i) = error_val(i) + error_val_val;
	%error_train(i) = error_train(i) + error_train_val;
 end
 figure(1)
 plot(1:m, error_train, 1:m, error_val);
 title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
 xlabel('Number of training examples')
 ylabel('Error')
 axis([0 13 0 100])
 legend('Train', 'Cross Validation')
 fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
 fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
 	for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));	
	end
%%-------------------------------------------------------------------------

% Plot training data and fit
figure(2);
load ('ex5data1.mat');
plot(X, y, 'rx', 'MarkerSize', 2, 'LineWidth', 1);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));