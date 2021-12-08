clear;
clc;

datasetDir = 'D:\Ahsen_Thesis\Weizman';
vidPaths = dir(datasetDir);

videoPlayer = vision.VideoPlayer();
for i = 3 : size(vidPaths,1)
    folderName = strcat(datasetDir,'\', vidPaths(i).name);
    videosPaths = dir(folderName);
    fprintf('Processing %s folder\n',folderName);
    for j = 3 : size(videosPaths,1)    
        vidName = strcat(datasetDir,'\', vidPaths(i).name, '\', videosPaths(j).name);        
        videoSource = vision.VideoFileReader(vidName, 'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
        ind = 1;
        folderSavePath = strcat(strrep(datasetDir,'Weizman','WeizmanFrames'), '\', vidPaths(i).name, '\', videosPaths(j).name);
        mkdir(folderSavePath);
        while ~isDone(videoSource)
            frame  = step(videoSource);
            frameSavePath = strcat(folderSavePath,'\',num2str(ind),'.jpg');
            imwrite(frame,frameSavePath);
            ind = ind + 1;
        end
    end
end