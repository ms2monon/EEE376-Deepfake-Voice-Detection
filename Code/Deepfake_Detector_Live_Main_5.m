% =============================================================
% Deepfake_Detector_Live_Main_5.m
% Step 5 – Live Deepfake Voice Detector
%
% Options:
%   1 = Record from microphone (4 seconds)
%   2 = Select an existing audio file
%   0 = Exit
%
% Requires: best_rf_model.mat   (from Classifier_4.m)
%           audioPreprocess.m   (must be in same folder)
% =============================================================

clc; close all;

% Load model
if ~isfile('best_rf_model.mat')
    error('best_rf_model.mat not found. Run Classifier_4.m first.');
end
load('best_rf_model.mat','rfModel','nFeat');
fprintf('Model loaded. Expects %d features.\n\n', nFeat);

Fs_target   = 16000;
nMfcc       = 13;
recDuration = 4;

% =============================================================
% Detection loop
% =============================================================
while true

    fprintf('=== Deepfake Detector ===\n');
    fprintf('  1 = Record from microphone (%d s)\n', recDuration);
    fprintf('  2 = Select audio file\n');
    fprintf('  0 = Exit\n');
    choice = input('Choice: ');

    if choice == 0, break; end

    fullPath = '';

    if choice == 1
        fprintf('\nSpeak now – recording for %d seconds...\n', recDuration);
        rec = audiorecorder(Fs_target, 16, 1);
        recordblocking(rec, recDuration);
        disp('Recording done.');
        y_raw    = getaudiodata(rec,'double');
        fullPath = 'temp_recording.wav';
        audiowrite(fullPath, y_raw, Fs_target);

    elseif choice == 2
        [file, path] = uigetfile({'*.wav;*.mp3;*.m4a','Audio Files'}, ...
                                  'Select Audio File');
        if isequal(file,0), disp('Cancelled.'); continue; end
        fullPath = fullfile(path, file);
        fprintf('File: %s\n', fullPath);

    else
        disp('Invalid choice.'); continue;
    end

    try
        % Load + preprocess
        [y_raw, fs] = audioread(fullPath);
        y = audioPreprocess(y_raw, fs);

        if length(y) < 4000
            disp('Audio too short after processing. Try again.'); continue;
        end

        % Extract features
        feat = extractFeatures(y, Fs_target, nMfcc);

        % Safety: pad or trim if dimension mismatch
        n = nFeat;
        if length(feat) < n,     feat(end+1:n) = 0;
        elseif length(feat) > n, feat = feat(1:n); end

        % Predict
        [predCell, scores] = predict(rfModel, feat);
        predLabel = str2double(predCell{1});

        classNames = str2double(rfModel.ClassNames);
        fakeCol    = find(classNames == 1);
        realCol    = find(classNames == 0);
        fakeProb   = scores(fakeCol);
        realProb   = scores(realCol);

        if predLabel == 1
            verdict  = 'FAKE / AI-GENERATED VOICE';
            barColor = [0.85 0.15 0.15];
        else
            verdict  = 'REAL HUMAN VOICE';
            barColor = [0.10 0.65 0.25];
        end

        % Console result
        fprintf('\n========================================\n');
        fprintf('  Verdict  : %s\n', verdict);
        fprintf('  Real     : %.1f%%\n', realProb*100);
        fprintf('  Fake     : %.1f%%\n', fakeProb*100);
        fprintf('========================================\n\n');

        % Visualisation figure
        figure('Name',verdict,'NumberTitle','off','Position',[80 80 1100 620]);

        subplot(2,2,1);
        t = (0:length(y)-1)/Fs_target;
        plot(t,y,'Color',[0.2 0.4 0.8]); grid on;
        title('Waveform'); xlabel('Time (s)'); ylabel('Amplitude');

        subplot(2,2,2);
        spectrogram(y,hamming(512),256,512,Fs_target,'yaxis','MinThreshold',-80);
        title('Spectrogram'); colorbar;

        subplot(2,2,3);
        bh = bar([realProb fakeProb]*100,'FaceColor','flat');
        bh.CData(1,:) = [0.10 0.65 0.25];
        bh.CData(2,:) = [0.85 0.15 0.15];
        set(gca,'XTickLabel',{'Real','Fake'});
        ylabel('Probability (%)'); ylim([0 108]);
        title('Prediction Confidence'); grid on;
        for k = 1:2
            vals = [realProb fakeProb]*100;
            text(k, vals(k)+2, sprintf('%.1f%%',vals(k)), ...
                 'HorizontalAlignment','center','FontWeight','bold');
        end

        subplot(2,2,4);
        imp = rfModel.OOBPermutedPredictorDeltaError;
        [sImp,sIdx] = sort(imp,'descend');
        nShow = min(10,length(imp));
        barh(sImp(1:nShow),'FaceColor',barColor);
        set(gca,'YTick',1:nShow, ...
            'YTickLabel',arrayfun(@(k) sprintf('F%d',sIdx(k)), ...
            1:nShow,'UniformOutput',false),'YDir','reverse');
        xlabel('Importance'); title('Top-10 Features'); grid on;

        sgtitle(sprintf('Analysis:  %s', verdict),'FontSize',13);

    catch ME
        fprintf('Error: %s\n', ME.message);
    end

    input('Press Enter to continue...','s');
    clc;
end

disp('Detector closed.');

% =============================================================
% Feature extraction — must match Feature_Extraction_2.m exactly
% =============================================================
function feat = extractFeatures(y, Fs, nMfcc)
    coeff = mfcc(y, Fs);
    if size(coeff,2)>nMfcc, coeff=coeff(:,1:nMfcc); end
    delta = diff(coeff); delta=[delta(1,:);delta];
    dd    = diff(delta);  dd=[dd(1,:);dd];
    mfccFeat=[mean(coeff,1) std(coeff,0,1) ...
              mean(delta,1) std(delta,0,1) ...
              mean(dd,1)    std(dd,0,1)];

    frameLen=round(0.025*Fs); hopLen=round(0.010*Fs);
    nFFT=2^nextpow2(frameLen); win=hann(frameLen,'periodic');
    frames=buffer(y,frameLen,frameLen-hopLen,'nodelay');
    frames=bsxfun(@times,frames,win);
    S=abs(fft(frames,nFFT)).^2; S=S(1:nFFT/2+1,:);
    freqs=(0:nFFT/2)'*Fs/nFFT;
    zcr=sum(abs(diff(sign(frames),1,1)),1)/(2*frameLen);
    rmsE=sqrt(mean(frames.^2,1));
    Snorm=S./(sum(S,1)+eps);
    sc=freqs'*Snorm;
    sb=sqrt(sum((freqs-sc).^2.*Snorm,1));
    nFr=size(frames,2); cumS=cumsum(S,1);
    thresh=0.85*sum(S,1); sr=zeros(1,nFr);
    for f=1:nFr
        idx=find(cumS(:,f)>=thresh(f),1,'first');
        if ~isempty(idx), sr(f)=freqs(idx); end
    end
    specFeat=[mean(zcr) std(zcr) mean(rmsE) std(rmsE) ...
              mean(sc)  std(sc)  mean(sb)   std(sb) ...
              mean(sr)  std(sr)];
    feat=[mfccFeat specFeat];
end
