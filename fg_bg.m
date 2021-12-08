clear;
clc;

datasetDir = 'D:\Ahsen_Thesis\KTH';
vidPaths = dir(datasetDir);
blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 120);
   
shapeInserter = vision.ShapeInserter('BorderColor','White');
peopleDetector = vision.PeopleDetector;
optical = opticalFlowLK('NoiseThreshold',0.009);
videoPlayer = vision.VideoPlayer();
for i = 3 : size(vidPaths,1)
    folderName = strcat(datasetDir,'\', vidPaths(i).name);
    videosPaths = dir(folderName);
    fprintf('Processing %s folder\n',folderName);
    detector = vision.ForegroundDetector('NumGaussians', 5, 'NumTrainingFrames', 25, 'MinimumBackgroundRatio', 0.7);
    for j = 3 : size(videosPaths,1)-98    
        vidName = strcat(datasetDir,'\', vidPaths(i).name, '\', videosPaths(j).name);
        videoSource = vision.VideoFileReader(vidName, 'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
%         ind = 1;
%         folderSavePath = strcat(strrep(datasetDir,'Weizman','WeizmanFrames'), '\', vidPaths(i).name, '\', videosPaths(j).name);
%         mkdir(folderSavePath);
        while ~isDone(videoSource)
            frame  = step(videoSource);
            fgMask = step(detector, frame);
%             frameSavePath = strcat(folderSavePath,'\',num2str(ind),'.jpg');
%             imwrite(frame,frameSavePath);
%             ind = ind + 1;
%             bbox   = step(blob, fgMask);
%             [bboxes,scores] = step(peopleDetector,frame);
            mask = estimateFlow(optical,frame);
            %             out    = step(shapeInserter, fgMask, bboxes);
            out = logical(mask.Magnitude>0.1);
%             out = imopen(out, strel('rectangle', [3,3]));
%             out = imclose(out, strel('disk', 15));
            step(videoPlayer, out);
            pause(0.02);
        end
    end
end