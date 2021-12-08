
clear;
clc;
close all;

datasetDir = '/media/itu/E6540E96540E699F/Thesis/KTHFrames';
trainingDir = strcat(datasetDir,'/', 'training');
testingDir = strcat(datasetDir, '/', 'testing');
fprintf('Loading Data \n');
if(~exist('KTH_Data.mat', 'file'))
    trainImagesPaths = imageSet(trainingDir,'recursive');
    testImagesPaths = imageSet(testingDir,'recursive');
    categories = {'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'};
    fprintf('Loading Training Data \n');
    [XTrain, tTrain] = readDataset(trainImagesPaths,categories);
    fprintf('Loading Testing Data \n');
    [xtest, tTest] = readDataset(testImagesPaths,categories);
    XTest = zeros(size(xtest{1},1)*size(xtest{1},2), size(xtest,1));
    for z = 1 : size(xtest,1)
        XTest(:,z) = reshape(xtest{z},[size(xtest{1},1)*size(xtest{1},2), 1]);
    end
    clear xtest;
    save KTH_Data.mat XTrain XTest tTrain tTest
else
    load KTH_Data.mat
end

fprintf('Training 1st AutoEncoder \n');

hiddenSize1 = 1000;
autoenc1 = trainAutoencoder(XTrain,hiddenSize1, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

feat1 = encode(autoenc1, XTrain);

fprintf('Training 2nd AutoEncoder \n');

hiddenSize2 = 700;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

fprintf('Training 3rd AutoEncoder \n');

hiddenSize3 = 250;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat3 = encode(autoenc3,feat2);

fprintf('Training 4th AutoEncoder \n');

hiddenSize4 = 100;
autoenc4 = trainAutoencoder(feat3,hiddenSize4, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat4 = encode(autoenc4,feat3);

fprintf('Training 5th AutoEncoder \n');

hiddenSize5 = 25;
autoenc5 = trainAutoencoder(feat4,hiddenSize5, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat5 = encode(autoenc5,feat4);

fprintf('Training Softmax Classifier \n');
softnet = trainSoftmaxLayer(feat5,tTrain,'MaxEpochs',500);

fprintf('Stacking autoEncoders and softmax \n');
deepnet = stack(autoenc1,autoenc2,autoenc3, autoenc4, autoenc5, softnet);
view(deepnet)

fprintf('Testing Stacked AutoEncoder on Test Data \n');
y = deepnet(XTest);
figure,
plotconfusion(tTest,y);

fprintf('Fine tuning the complete stacked auto encoder network \n');
trainingData = zeros(size(XTrain{1},1)*size(XTrain{1},2), size(XTrain,1));
for z = 1 : size(XTrain,1)
    trainingData(:,z) = reshape(XTrain{z}, [size(XTrain{1},1)*size(XTrain{1},2), 1]);
end
clear XTrain;
deepnet = train(deepnet,trainingData,tTrain);
y = deepnet(XTest);
figure,
plotconfusion(tTest,y);
