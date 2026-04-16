function showModelMontage(modelPath, maxImages)
%   showModelMontage('trainedFaceModel.mat', 20)

if nargin < 1 || isempty(modelPath)
    modelPath = fullfile(fileparts(mfilename('fullpath')), 'trainedFaceModel.mat');
end

if nargin < 2 || isempty(maxImages)
    maxImages = 25;
end

if ~isfile(modelPath)
    error('Model file not found: %s', modelPath);
end

data = load(modelPath, 'model');
if ~isfield(data, 'model') || ~isfield(data.model, 'previewFaces')
    error('Model does not contain previewFaces. Re-run algoTraining.');
end

previewFaces = data.model.previewFaces;
if isempty(previewFaces)
    error('No preview faces available in model. Re-run algoTraining.');
end

if isfield(data.model, 'previewLabels')
    previewLabels = string(data.model.previewLabels);
else
    previewLabels = repmat("Face", numel(previewFaces), 1);
end

n = min(numel(previewFaces), maxImages);
numCols = 5;
numRows = ceil(n / numCols);

figure('Name', 'Registered Face Previews');
tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:n
    nexttile;
    imshow(previewFaces{i});
    title(previewLabels(i), 'Interpreter', 'none', 'FontSize', 10);
end

sgtitle(sprintf('Face previews from model (%d shown)', n));
end
