% =============================================================
% Feature_Extraction_2.m  —  DEBUG VERSION
% Run this to find exactly which file and which line causes
% "Index exceeds number of array elements"
% =============================================================

clc;

if ~exist('T','var') || isempty(T)
    if isfile('dataset_table.mat')
        load('dataset_table.mat', 'T');
        fprintf('Loaded dataset_table.mat\n');
    else
        error('dataset_table.mat not found. Run Build_Dataset_Table_1.m first.');
    end
end

Fs_target  = 16000;
nMfcc      = 13;
nFeatTotal = 88;
minSamples = Fs_target * 1;

nFiles        = height(T);
featureMatrix = nan(nFiles, nFeatTotal);

fprintf('Extracting features from %d files...\n\n', nFiles);

for i = 1:nFiles
    filepath = T.FilePath{i};
    fprintf('[%d/%d] %s\n', i, nFiles, filepath);

    try
        % A: read
        [y_raw, fs] = audioread(filepath);
        fprintf('  A: size=%dx%d fs=%d\n', size(y_raw,1),size(y_raw,2),fs);

        % B: preprocess
        y = audioPreprocess(y_raw, fs);
        fprintf('  B: after preprocess length=%d (%.2fs)\n', ...
                length(y), length(y)/Fs_target);

        % C: length check
        if length(y) < minSamples
            fprintf('  C: SKIP too short\n\n');
            continue;
        end

        % D: features
        feat = extractFeatures(y, Fs_target, nMfcc);
        fprintf('  D: feat length=%d\n', length(feat));

        featureMatrix(i, 1:min(length(feat),nFeatTotal)) = feat(1:min(length(feat),nFeatTotal));
        fprintf('  DONE\n\n');

    catch ME
        fprintf('  *** ERROR ***\n');
        fprintf('  %s\n', ME.message);
        for k = 1:length(ME.stack)
            fprintf('  Line %d in %s\n', ME.stack(k).line, ME.stack(k).name);
        end
        fprintf('\n');
    end
end

validRows     = ~any(isnan(featureMatrix),2);
featureMatrix = featureMatrix(validRows,:);
labelsClean   = T.Label(validRows);

fprintf('Valid: %d / %d\n', sum(validRows), nFiles);
fprintf('Matrix: %d x %d\n', size(featureMatrix));

save('feature_matrix_final_safe.mat', 'featureMatrix');
save('labels_clean_final.mat',        'labelsClean');
fprintf('Saved.\n');

% =============================================================
function feat = extractFeatures(y, Fs, nMfcc)
    feat = nan(1,88);

    fprintf('  [feat] length(y)=%d\n', length(y));

    if length(y) < Fs*0.5, return; end

    % MFCC
    try
        coeff = mfcc(y, Fs);
    catch ME2
        fprintf('  [feat] mfcc failed: %s\n', ME2.message);
        return;
    end
    fprintf('  [feat] coeff size=%dx%d\n', size(coeff,1),size(coeff,2));

    if size(coeff,2) > nMfcc, coeff = coeff(:,1:nMfcc); end
    if size(coeff,1) < 2, return; end

    delta = diff(coeff); delta = [delta(1,:); delta];
    dd    = diff(delta);  dd   = [dd(1,:);    dd];
    mfccFeat = [mean(coeff,1) std(coeff,0,1) ...
                mean(delta,1) std(delta,0,1) ...
                mean(dd,1)    std(dd,0,1)];

    % Spectral
    frameLen = round(0.025*Fs);
    hopLen   = round(0.010*Fs);
    nFFT     = 2^nextpow2(frameLen);
    fprintf('  [feat] frameLen=%d hopLen=%d nFFT=%d length(y)=%d\n', ...
            frameLen, hopLen, nFFT, length(y));

    if length(y) < frameLen*2
        feat = [mfccFeat nan(1,10)];
        return;
    end

    win    = hann(frameLen,'periodic');
    frames = buffer(y, frameLen, frameLen-hopLen, 'nodelay');
    fprintf('  [feat] frames=%dx%d\n', size(frames,1),size(frames,2));

    if size(frames,1) < frameLen
        frames(end+1:frameLen,:) = 0;
    end
    nFrames = size(frames,2);
    if nFrames < 1
        feat = [mfccFeat nan(1,10)]; return;
    end

    frames = bsxfun(@times, frames, win);
    S      = abs(fft(frames,nFFT)).^2;
    S      = S(1:nFFT/2+1,:);
    freqs  = (0:nFFT/2)'*Fs/nFFT;
    fprintf('  [feat] S=%dx%d freqs=%d nFrames=%d\n', ...
            size(S,1),size(S,2),length(freqs),nFrames);

    zcr    = sum(abs(diff(sign(frames),1,1)),1)/(2*frameLen);
    rmsE   = sqrt(mean(frames.^2,1));
    Snorm  = S./(sum(S,1)+eps);
    sc     = freqs'*Snorm;
    sb     = sqrt(sum((freqs-sc).^2.*Snorm,1));
    cumS   = cumsum(S,1);
    thresh = 0.85*sum(S,1);
    sr     = zeros(1,nFrames);
    for f  = 1:nFrames
        idx = find(cumS(:,f)>=thresh(f),1,'first');
        if ~isempty(idx), sr(f)=freqs(idx); end
    end

    specFeat=[mean(zcr) std(zcr) mean(rmsE) std(rmsE) ...
              mean(sc)  std(sc)  mean(sb)   std(sb) ...
              mean(sr)  std(sr)];

    feat=[mfccFeat specFeat];
end
