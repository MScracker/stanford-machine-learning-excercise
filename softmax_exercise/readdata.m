clear all,close all,clc;
filename = 'mnist/train-images-idx3-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

numImages = numImages/2; 
images1 = fread(fp, numRows * numCols * numImages, 'unsigned char');
images2 = fread(fp, numRows * numCols * numImages, 'unsigned char');
fclose(fp);
images1 = reshape(images1, numRows * numCols, numImages);
images2 = reshape(images2, numRows * numCols, numImages);
images = [images1;images2];
disp('reading data is over');