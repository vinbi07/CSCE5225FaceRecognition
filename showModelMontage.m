function showModelMontage(modelPath, maxImages)

% Input Arguments
if nargin < 1 || isempty(modelPath)
    modelPath = fullfile(fileparts(mfilename('fullpath')), 'trainedFaceModel.mat');
end

if nargin < 2 || isempty(maxImages)
    maxImages = inf;
end

if ~isfile(modelPath)
    error('Model file not found: %s', modelPath);
end

% Get Data from Training
data = load(modelPath, 'model');
if ~isfield(data, 'model') || ~isfield(data.model, 'previewFaces')
    error('Model does not contain previewFaces. Re-run algoTraining.');
end

% Sync labels and faces
previewFaces = data.model.previewFaces;
if isempty(previewFaces)
    error('No preview faces available in model. Re-run algoTraining.');
end

if isfield(data.model, 'previewLabels')
    previewLabels = string(data.model.previewLabels);
else
    previewLabels = repmat("Face", numel(previewFaces), 1);
end

% Labels and face same length
if numel(previewLabels) < numel(previewFaces)
    previewLabels(end + 1:numel(previewFaces), 1) = "Face";
elseif numel(previewLabels) > numel(previewFaces)
    previewLabels = previewLabels(1:numel(previewFaces));
end

% Get the names accounting for augmentation stuff
personNames = previewLabels;
hasAugmentationSuffix = contains(previewLabels, " | ");
personNames(hasAugmentationSuffix) = extractBefore(previewLabels(hasAugmentationSuffix), " | ");
uniquePeople = unique(personNames, 'stable');

if isempty(uniquePeople)
    error('No person labels available in model. Re-run algoTraining.');
end

% UI to display what page of registered faces is being shown
fig = figure( ...
    'Name', 'Registered Face Previews', ...
    'NumberTitle', 'off');
setappdata(fig, 'currentPersonIndex', 1);

% Page Buttons
uicontrol( ...
    'Style', 'pushbutton', ...
    'String', 'Previous', ...
    'Position', [10 10 90 30], ...
    'Callback', @showPreviousPerson);

uicontrol( ...
    'Style', 'pushbutton', ...
    'String', 'Next', ...
    'Position', [110 10 90 30], ...
    'Callback', @showNextPerson);

renderCurrentPerson();

% Function to render the current person's faces
    function renderCurrentPerson()
        currentPersonIndex = getappdata(fig, 'currentPersonIndex');
        currentPerson = uniquePeople(currentPersonIndex);
        currentMask = personNames == currentPerson;

        personFaces = previewFaces(currentMask);
        personLabels = previewLabels(currentMask);

        n = min(numel(personFaces), maxImages);
        numCols = min(4, max(1, ceil(sqrt(n))));
        numRows = ceil(n / numCols);

        clf(fig);

        uicontrol( ...
            'Style', 'pushbutton', ...
            'String', 'Previous', ...
            'Position', [10 10 90 30], ...
            'Callback', @showPreviousPerson);

        uicontrol( ...
            'Style', 'pushbutton', ...
            'String', 'Next', ...
            'Position', [110 10 90 30], ...
            'Callback', @showNextPerson);

        tiledlayout(fig, numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');

        for idx = 1:n
            nexttile;
            imshow(personFaces{idx});
            title(personLabels(idx), 'Interpreter', 'none', 'FontSize', 10);
        end

        sgtitle(sprintf('%s (%d of %d images shown) | Person %d of %d', ...
            currentPerson, n, numel(personFaces), currentPersonIndex, numel(uniquePeople)));
    end

% Functions for page buttons next/previous
    function showPreviousPerson(~, ~)
        currentPersonIndex = getappdata(fig, 'currentPersonIndex');
        currentPersonIndex = max(1, currentPersonIndex - 1);
        setappdata(fig, 'currentPersonIndex', currentPersonIndex);
        renderCurrentPerson();
    end

    function showNextPerson(~, ~)
        currentPersonIndex = getappdata(fig, 'currentPersonIndex');
        currentPersonIndex = min(numel(uniquePeople), currentPersonIndex + 1);
        setappdata(fig, 'currentPersonIndex', currentPersonIndex);
        renderCurrentPerson();
    end
end
