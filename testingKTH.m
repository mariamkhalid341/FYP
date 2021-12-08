
%this is the testing code for KTH dataset
load KTHTestingData.mat;
load networkKTH.mat;

%testing
view(deepnet);
y = deepnet(TestData);

%plot confusion matrix
figure,
plotconfusion(TestLabels,y);

