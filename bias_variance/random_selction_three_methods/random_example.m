%% Randomly Selected Samples

%% Initialization
clear all, close all, clc;

%% =========== Part 1: Loading and Visualizing Data =============
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

error_train = zeros(m,1);
error_val = zeros(m,1);
loops = 50;
% ==== Plotting learning curves with randomly selected examples ===
for i = 1 : loops
	for j = 1 : m
		%randomly select j examples from X
		sel = randperm(m);
		sel = sel(1:j);
		X_sel_train = X(sel,:);
		y_sel_train = y(sel,:);
		
		%Feature Mapping for Polynomial Regression 
		p = 8;
		% Map X onto Polynomial Features and Normalize
		X_poly_train = polyFeatures(X_sel_train, p);
		[X_poly_train, mu, sigma] = featureNormalize(X_poly_train);  % Normalize
		X_poly_train = [ones(size(X_poly_train,1), 1), X_poly_train];                   % Add Ones
		%  Train linear regression with lambda = 0
		fprintf('Training Linear Regression model...\n');
		lambda = 0.01;
		[theta] = trainLinearReg(X_poly_train, y_sel_train, lambda);
	
		%compute error_train(j)
		error_train_tmp = linearRegCostFunction(X_poly_train, y_sel_train, theta, 0);
		error_train(j) = error_train(j) + error_train_tmp;
		
		%randomly select j examples from Xval
		sel = randperm(m);
		sel = sel(1:j);
		X_sel_val = Xval(sel,:);
		y_sel_val = yval(sel,:);
		% Map X_poly_val and normalize (using mu and sigma)
		X_poly_val = polyFeatures(X_sel_val, p);
		X_poly_val = bsxfun(@minus, X_poly_val, mu);
		X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
		X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones
		% compute error_val(j)
		error_val_tmp = linearRegCostFunction(X_poly_val, y_sel_val, theta, 0);
		error_val(j) = error_val(j) + error_val_tmp;
		
	end

end
	error_train = error_train./loops;
	error_val = error_val./loops;
	
	%print error_train and error_val
	fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
	for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));	
	end
	
	%plot learning curves
	figure(1)
	plot(1:m, error_train, 1:m, error_val);
	title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
	xlabel('Number of training examples')
	ylabel('Error')
	axis([0 13 0 100])
	legend('Train', 'Cross Validation')
	
	% Plot training data and fit
	figure(2)
	plot(X, y, 'rx', 'MarkerSize', 2, 'LineWidth', 1);
	plotFit(min(X), max(X), mu, sigma, theta, p);
	xlabel('Change in water level (x)');
	ylabel('Water flowing out of the dam (y)');
	title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));