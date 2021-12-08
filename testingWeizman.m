
%this is the testing code for Weizman dataset
load WeizmanTestingData.mat;
load networkWeizman.mat;

%testing
view(deepnet);
y = deepnet(TestData);

%plot confusion matrix
figure,
plotconfusion(TestLabels,y);

