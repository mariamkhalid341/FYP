clear;
clc;

datasetDir = 'D:\ML_Server\KTH Dataset\KTH';
vidPaths = dir(datasetDir);
%object for optical flow (Lucas Kanade Method)
optical = opticalFlowLK('NoiseThreshold',0.009);
%object for video player
videoPlayer = vision.VideoPlayer();
%threshold for optical magnitude
th = 0.1;
rows = 5;
window = 15;
for i = 3 : size(vidPaths,1)
    folderName = strcat(datasetDir,'\', vidPaths(i).name);
    videosPaths = dir(folderName);
    fprintf('Processing %s folder\n',folderName);
    for j = 3 : size(videosPaths,1)    
        vidName = strcat(datasetDir,'\', vidPaths(i).name, '\', videosPaths(j).name);        
        videoSource = vision.VideoFileReader(vidName, 'ImageColorSpace','Intensity','VideoOutputDataType','uint8');
        ind = 1;
        folderSavePath = strcat(strrep(datasetDir,'KTH','KTHFrames'), '\', vidPaths(i).name, '\', videosPaths(j).name);
        mkdir(folderSavePath);
        while ~isDone(videoSource)
            frame  = step(videoSource);
            mask = estimateFlow(optical,frame);
            out = logical(mask.Magnitude>th);
            [r, c] = size(out);
            out(1:rows,:) = 0;
            out(end-rows:end,:) = 0;
            out(:,1:rows) = 0;
            out(:,end-rows:end) = 0;
            colSum = sum(out);
            if(sum(out(:))>20)
                indices = find(colSum);
                first = indices(1);
                last = indices(end);
                if(first - window<=0)
                    first = 1;
                else
                    first = first-window;
                end
                if(last + window>=c)
                    last = c;
                else
                    last = last + window;
                end
                outputImg = zeros(r,c);
                outputImg(:, first:last) = frame(:,first:last);

                frameSavePath = strcat(folderSavePath,'\',num2str(ind),'.jpg');
                imwrite(uint8(outputImg),frameSavePath);
            end
            ind = ind + 1;
        end
    end
end