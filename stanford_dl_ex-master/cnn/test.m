imageDim = size(db_images,1); % height/width of image
numImages = size(db_images,3); % number of db_images

%% Reshape parameters and setup gradient matrices

% Wc is db_filterDim x db_filterDim x db_numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(db_theta,imageDim,db_filterDim,db_numFilters,...
                        db_poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-db_filterDim+1; % dimension of convolved output
outputDim = (convDim)/db_poolDim; % dimension of subsampled output

% convDim x convDim x db_numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,db_numFilters,numImages);

% outputDim x outputDim x db_numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,db_numFilters,numImages);

%%% YOUR CODE HERE %%%
activations = cnnConvolve(db_filterDim, db_numFilters, db_images, Wc, bc);
activationsPooled = cnnPool(db_poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);