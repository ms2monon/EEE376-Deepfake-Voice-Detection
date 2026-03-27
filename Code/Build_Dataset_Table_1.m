% =============================================================
% Build_Dataset_Table_1.m
% Step 1 – Dataset Setup
% Scans your real/ and fake/ folders, builds dataset_table.mat
% Run this FIRST before anything else.
% =============================================================

clc; clear; close all;

% -------------------------------------------------------------
% Dataset root path
% -------------------------------------------------------------
datasetRoot = 'E:\EEE 376 MATLAB\EEE 376 Project\Dataset for EEE_376\WAV_converted';

realDir = fullfile(datasetRoot, 'real');
fakeDir = fullfile(datasetRoot, 'fake');

if ~isfolder(realDir) || ~isfolder(fakeDir)
    error(['Could not find real/ or fake/ subfolders.\n' ...
           'Expected:\n  %s\n  %s\n' ...
           'Please check that datasetRoot is correct.'], ...
           realDir, fakeDir);
end

% -------------------------------------------------------------
% Collect all WAV files
% -------------------------------------------------------------
audioExts = {'*.wav','*.mp3','*.m4a','*.flac'};

fprintf('Scanning folders...\n');
realFiles = collectFiles(realDir, audioExts);
fakeFiles = collectFiles(fakeDir, audioExts);

fprintf('  Real files found: %d\n', numel(realFiles));
fprintf('  Fake files found: %d\n', numel(fakeFiles));

if numel(realFiles) == 0 || numel(fakeFiles) == 0
    error('No audio files found. Check your folder paths and file extensions.');
end

% -------------------------------------------------------------
% Build dataset table   Label: 0 = real,  1 = fake
% -------------------------------------------------------------
allPaths  = [realFiles; fakeFiles];
allLabels = [zeros(numel(realFiles),1,'int32'); ...
              ones(numel(fakeFiles), 1,'int32')];

rng(42);
idx = randperm(numel(allPaths));
T   = table(allPaths(idx), allLabels(idx), ...
            'VariableNames', {'FilePath','Label'});

fprintf('\nTotal samples: %d  (real=%d, fake=%d)\n', ...
        height(T), sum(T.Label==0), sum(T.Label==1));

% -------------------------------------------------------------
% Quick sanity check on first 5 files
% -------------------------------------------------------------
fprintf('\nSanity check (first 5 files):\n');
for k = 1:min(5, height(T))
    try
        info = audioinfo(T.FilePath{k});
        fprintf('  [%d] %.2fs  %dHz  %s\n', ...
                k, info.Duration, info.SampleRate, T.FilePath{k});
    catch
        fprintf('  [%d] ERROR reading: %s\n', k, T.FilePath{k});
    end
end

% -------------------------------------------------------------
% Save
% -------------------------------------------------------------
save('dataset_table.mat', 'T');
fprintf('\ndataset_table.mat saved.\n');
fprintf('Next: run Feature_Extraction_2.m\n');

% =============================================================
function files = collectFiles(folder, exts)
    files = {};
    for e = 1:numel(exts)
        found = dir(fullfile(folder, exts{e}));
        for f = 1:numel(found)
            files{end+1,1} = fullfile(found(f).folder, found(f).name);
        end
    end
end
