% =============================================================
% Deepfake_Detector_Live_Main_6.m
% Step 6 – Live Deepfake Voice Detector (SVM version)
%
% Options:
%   1 = Record from microphone (4 seconds)
%   2 = Select an existing audio file
%   0 = Exit
%
% Requires: best_svm_model.mat  (from Classifier_4.m)
%           audioPreprocess.m   (must be in same folder)
% =============================================================

clc; close all;

% Load SVM model
if ~isfile('best_svm_model.mat')
    error('best_svm_model.mat not found. Run Classifier_4.m first.');
end
load('best_svm_model.mat', 'svmModel');
fprintf('SVM model loaded.\n\n');

Fs_target   = 16000;
nMfcc       = 13;
nFeat       = 88;
recDuration = 4;

% =============================================================
% Detection loop
% =============================================================
while true

    fprintf('=== Deepfake Detector (SVM) ===\n');
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
        y_raw    = getaudiodata(rec, 'double');
        fullPath = 'temp_recording.wav';
        audiowrite(fullPath, y_raw, Fs_target);

    elseif choice == 2
        [file, path] = uigetfile({'*.wav;*.mp3;*.m4a;*.flac','Audio Files'}, ...
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
        if length(feat) < nFeat,     feat(end+1:nFeat) = 0;
        elseif length(feat) > nFeat, feat = feat(1:nFeat); end

        % Predict using SVM
        [predLabel, scores] = predict(svmModel, feat);
        predLabel = double(predLabel);

        % Get probability for each class
        classNames  = svmModel.ClassNames;
        fakeCol     = find(classNames == 1);
        realCol     = find(classNames == 0);
        fakeProb    = scores(fakeCol);
        realProb    = scores(realCol);

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
        fprintf('  Model    : SVM (RBF kernel)\n');
        fprintf('========================================\n\n');

        % Visualisation figure
        figure('Name', verdict, 'NumberTitle','off', 'Position',[80 80 1100 620]);

        % Waveform
        subplot(2,2,1);
        t = (0:length(y)-1) / Fs_target;
        plot(t, y, 'Color',[0.2 0.4 0.8]); grid on;
        title('Waveform'); xlabel('Time (s)'); ylabel('Amplitude');

        % Spectrogram
        subplot(2,2,2);
        spectrogram(y, hamming(512), 256, 512, Fs_target, 'yaxis', 'MinThreshold',-80);
        title('Spectrogram'); colorbar;

        % Confidence bar
        subplot(2,2,3);
        bh = bar([realProb fakeProb]*100, 'FaceColor','flat');
        bh.CData(1,:) = [0.10 0.65 0.25];
        bh.CData(2,:) = [0.85 0.15 0.15];
        set(gca, 'XTickLabel',{'Real','Fake'});
        ylabel('Probability (%)'); ylim([0 108]);
        title('Prediction Confidence'); grid on;
        for k = 1:2
            vals = [realProb fakeProb]*100;
            text(k, vals(k)+2, sprintf('%.1f%%', vals(k)), ...
                 'HorizontalAlignment','center','FontWeight','bold');
        end

        % Probability gauge (replaces feature importance — SVM has none)
        subplot(2,2,4);
        theta  = linspace(pi, 0, 200);
        xArc   = cos(theta);
        yArc   = sin(theta);
        fill([0 xArc 0], [0 yArc 0], [0.93 0.93 0.93], 'EdgeColor','none');
        hold on;

        % Filled arc up to fakeProb
        thetaFill = linspace(pi, pi - fakeProb*pi, 100);
        xF = cos(thetaFill); yF = sin(thetaFill);
        fill([0 xF 0], [0 yF 0], barColor, 'EdgeColor','none', 'FaceAlpha',0.85);

        % Needle
        angle  = pi - fakeProb*pi;
        plot([0 0.75*cos(angle)],[0 0.75*sin(angle)], ...
             'k-','LineWidth',2.5);
        plot(0, 0, 'ko', 'MarkerSize',8, 'MarkerFaceColor','k');

        % Labels
        text(-1.05, 0, 'Real', 'FontSize',11, 'HorizontalAlignment','right', ...
             'Color',[0.10 0.65 0.25], 'FontWeight','bold');
        text( 1.05, 0, 'Fake', 'FontSize',11, 'HorizontalAlignment','left', ...
             'Color',[0.85 0.15 0.15], 'FontWeight','bold');
        text(0, 0.35, sprintf('%.1f%%\nFake', fakeProb*100), ...
             'HorizontalAlignment','center','FontSize',14,'FontWeight','bold', ...
             'Color', barColor);

        axis equal; axis off;
        title('Fake Probability Gauge', 'FontSize',11);
        hold off;

        sgtitle(sprintf('SVM Analysis:  %s', verdict), 'FontSize',13);

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
