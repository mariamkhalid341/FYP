function[TrainImages, i] = readvideos(path, s, e)

videos = dir(strcat(path,'*.avi'));

detector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 15, 'MinimumBackgroundRatio', 0.7);
i=1;
for k=s : e
    filename = strcat(path, videos(k).name);
    videoSource = vision.VideoFileReader(filename, 'ImageColorSpace','Intensity','VideoOutputDataType','uint8');

    while ~isDone(videoSource)
        frame  = imresize((step(videoSource)), [64 64]);
%         fgMask = step(detector, frame);
%         TrainImages{i} = fgMask;
        TrainImages{i} = frame;
%         tTest(i) = 0;
        i = i+1;
    end
end