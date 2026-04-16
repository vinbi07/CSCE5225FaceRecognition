function outFrame = preProcess(inFrame, targetSize)
% Addison Nally
% preProcess.m
% Simple preprocessing for face detection

if nargin < 1 || isempty(inFrame)
    error('preProcess requires one input frame.');
end

if nargin < 2
    targetSize = [200 200];
end

if size(inFrame, 3) == 3
    grayFrame = rgb2gray(inFrame);
else
    grayFrame = inFrame;
end

% Reduce Noise
% Spacial Filtering
% Fouirier Transform
% Canny Edge Detection (Edge Detection)


eqFrame = histeq(grayFrame);
resizedFrame = imresize(eqFrame, targetSize);
outFrame = medfilt2(resizedFrame, [3 3]);
end