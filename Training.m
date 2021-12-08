clear 
clc

path = ('D:\Ahsen_Thesis\Weizman\bend\');
path1 = ('D:\Ahsen_Thesis\Weizman\jack\');
path2 = ('D:\Ahsen_Thesis\Weizman\jump\');
path3 = ('D:\Ahsen_Thesis\Weizman\pjump\');
path4 = ('D:\Ahsen_Thesis\Weizman\run\');
path5 = ('D:\Ahsen_Thesis\Weizman\side\');
path6 = ('D:\Ahsen_Thesis\Weizman\skip\');
path7 = ('D:\Ahsen_Thesis\Weizman\walk\');
path8 = ('D:\Ahsen_Thesis\Weizman\wave1\');
path9 = ('D:\Ahsen_Thesis\Weizman\wave2\');

trainStartInd = 1;
trainEndInd = 5;
testStartInd = 6;
testEndInd = 9;

classes = 10;

i=1;
Images = readvideos(path, trainStartInd, trainEndInd);
TrainImages = Images;
tTrain(1:length(Images)) = 0;

Images = readvideos(path1, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 1;

Images = readvideos(path2, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 2;

Images = readvideos(path3, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 3;

Images = readvideos(path4, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 4;

Images = readvideos(path5, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 5;

Images = readvideos(path6, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 6;

Images = readvideos(path7, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 7;

Images = readvideos(path8, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 8;

Images = readvideos(path9, trainStartInd, trainEndInd);
TrainImages(end+1 : end+length(Images)) = Images;
tTrain(end+1 : end+length(Images)) = 9;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Images = readvideos(path, testStartInd, testEndInd);
TestImages = Images;
tTest(1:length(Images)) = 0;

Images = readvideos(path1, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 1;

Images = readvideos(path2, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 2;

Images = readvideos(path3, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 3;

Images = readvideos(path4, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 4;

Images = readvideos(path5, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 5;

Images = readvideos(path6, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 6;

Images = readvideos(path7, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 7;

Images = readvideos(path8, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 8;

Images = readvideos(path9, testStartInd, testEndInd);
TestImages(end+1 : end+length(Images)) = Images;
tTest(end+1 : end+length(Images)) = 9;

tTrainTemp = tTrain;
tTestTemp = tTest;
tTrain = zeros(classes, size(tTrainTemp,2));
tTest = zeros(classes, size(tTestTemp,2));

for i = 1 : size(tTrainTemp,2)
    tTrain(tTrainTemp(i)+1,i) = 1;
end
for i = 1 : size(tTestTemp,2)
    tTest(tTestTemp(i)+1,i) = 1;
end

%training 1st AutoEncoder
fprintf('Training 1st AutoEncoder\n');
hiddenSize1 = 2048;
autoenc1 = trainAutoencoder(TrainImages,hiddenSize1, ...
    'MaxEpochs',50, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

figure()
plotWeights(autoenc1);

feat1 = encode(autoenc1,TrainImages);

%training 2nd AutoEncoder
fprintf('Training 2nd AutoEncoder\n');
hiddenSize2 = 500;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',20, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

figure()
plotWeights(autoenc2);

feat2 = encode(autoenc2,feat1);

%training 3rd AutoEncoder
fprintf('Training 3rd AutoEncoder\n');
hiddenSize3 = 200;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',10, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

% figure()
% plotWeights(autoenc3);

feat3 = encode(autoenc3,feat2);

%training Softmax Classification Layer
fprintf('Training Softmax Layer\n');
softnet = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',400);

%stacking autoencoders
deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);
view(deepnet)

% Get the number of pixels in each image
imageWidth = 64;
imageHeight = 64;
inputSize = imageWidth*imageHeight;


% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(TestImages));
for i = 1:numel(TestImages)
    xTest(:,i) = TestImages{i}(:);
end

y = deepnet(xTest);
plotconfusion(tTest,y);

%%%%%%%%%%%%%%%%%%%%%%%%%% Fine tuning the deep neural network %%%%%%%%%%%%%%%%%%%%%%%%%% 

% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(TrainImages));
for i = 1:numel(TrainImages)
    xTrain(:,i) = TrainImages{i}(:);
end

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);

y = deepnet(xTest);
plotconfusion(tTest,y);