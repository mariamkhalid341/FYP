clear
clc

[X,T] = iris_dataset;

hiddenSize = 5;
autoenc = trainAutoencoder(X, hiddenSize, ...
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'DecoderTransferFunction','purelin');

features = encode(autoenc,X);

softnet = trainSoftmaxLayer(features,T);

stackednet = stack(autoenc,softnet);

view(stackednet);