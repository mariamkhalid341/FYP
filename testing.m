clear;
clc;

datasetDir = 'D:\Ahsen_Thesis\WeizmanFrames';
imagesPaths = imageSet(datasetDir,'recursive');
categories = {'bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2'};
ind = 1;
for i = 1 : size(imagesPaths,2)
    for j = 1 : size(imagesPaths(i).ImageLocation,2)
        dataset{ind, 1} = imagesPaths(i).ImageLocation{j};
        if(~isempty(strfind(imagesPaths(i).Description, categories{1})))
            dataset{ind, 2} = categories{1};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{2})))
            dataset{ind, 2} = categories{2};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{4})))
            dataset{ind, 2} = categories{4};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{3})))
            dataset{ind, 2} = categories{3};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{5})))
            dataset{ind, 2} = categories{5};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{6})))
            dataset{ind, 2} = categories{6};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{7})))
            dataset{ind, 2} = categories{7};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{8})))
            dataset{ind, 2} = categories{8};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{9})))
            dataset{ind, 2} = categories{9};
        elseif(~isempty(strfind(imagesPaths(i).Description, categories{10})))
            dataset{ind, 2} = categories{10};
        end
        ind = ind + 1;
    end
end

ind = 1;

for i = 1 : size(dataset,1)
    if(strcmp(dataset{i,2},categories{1}))
        images{ind} = imresize(imread(dataset{i,1}),[32 32]);
        ind = ind + 1;
    elseif(strcmp(dataset{i,2},categories{5}))
        images{ind} = imresize(imread(dataset{i,1}),[32 32]);
        ind = ind + 1;
    end
end

XTrain(1:100) = images(1:100);
XTrain(101:200) = images(701:800);
target = zeros(2,100);
target(1,1:100) = 1;
target(2,101:200) = 1;

hiddenSize1 = 500;

autoenc1 = trainAutoencoder(XTrain,hiddenSize1, ...
    'MaxEpochs',500, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1)

feat1 = encode(autoenc1, XTrain);

hiddenSize2 = 256;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,target,'MaxEpochs',400);

deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)

ind = 1;

for i = 1 : size(dataset,1)
    if(strcmp(dataset{i,2},categories{1}))
        testImg(:,ind) = reshape(imresize(imread(dataset{i,1}),[32 32]),[1024 1]);
        ind = ind + 1;
    elseif(strcmp(dataset{i,2},categories{5}))
        testImg(:,ind) = reshape(imresize(imread(dataset{i,1}),[32 32]),[1024 1]);
        ind = ind + 1;
    end
end

XTest(:,1:100) = testImg(:,101:200);
XTest(:,101:200) = testImg(:,801:900);
tTest = zeros(2,100);
tTest(1,1:100) = 1;
tTest(2,101:200) = 1;

y = deepnet(XTest);
plotconfusion(tTest,y);

xTrain(:,1:100) = testImg(:,1:100);
xTrain(:,101:200) = testImg(:,701:800);
tTrain = zeros(2,100);
tTrain(1,1:100) = 1;
tTrain(2,101:200) = 1;

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);

y = deepnet(XTest);
plotconfusion(tTest,y);