% =============================================================
% Step 4 – Classifier Training & Evaluation
%
% Uses speaker-stratified split to prevent leakage:
%   all 6 files per student stay together in train OR test.
%
% Produces:
%   best_rf_model.mat, best_svm_model.mat
%   roc_curves.png, rf_feature_importance.png
%   per_speaker_accuracy.png
%
% Requires: feature_matrix_final_safe.mat
%           labels_clean_final.mat
%           dataset_table.mat
%           extractSpeakerIDs.m  (must be in same folder)
% =============================================================

clc;

% Load data
if ~exist('featureMatrix','var') || isempty(featureMatrix)
    if isfile('feature_matrix_final_safe.mat')
        load('feature_matrix_final_safe.mat', 'featureMatrix');
        load('labels_clean_final.mat',         'labelsClean');
        load('dataset_table.mat',              'T');
        fprintf('Loaded feature matrix.\n');
    else
        error('Run Feature_Extraction.m first.');
    end
end
labelsClean = double(labelsClean);  % fix int32 vs double type mismatch

nFeat = size(featureMatrix,2);
fprintf('\n=== Classifier Training ===\n');
fprintf('  Samples: %d   Features: %d\n\n', size(featureMatrix,1), nFeat);

% =============================================================
% 4.0  Extract speaker IDs
% =============================================================
speakerIDs     = extractSpeakerIDs(T.FilePath);
uniqueSpeakers = unique(speakerIDs);
nSpeakers      = numel(uniqueSpeakers);
fprintf('Unique speakers: %d\n', nSpeakers);

% =============================================================
% 4.1  Speaker-stratified 80/20 split
% =============================================================
rng(42);
shuffled      = uniqueSpeakers(randperm(nSpeakers));
nTest         = max(1, round(0.20 * nSpeakers));
testSpeakers  = shuffled(1:nTest);
trainSpeakers = shuffled(nTest+1:end);

trainMask = ismember(speakerIDs, trainSpeakers);
testMask  = ismember(speakerIDs, testSpeakers);

XTrain = featureMatrix(trainMask,:);  yTrain = labelsClean(trainMask);
XTest  = featureMatrix(testMask,:);   yTest  = labelsClean(testMask);

fprintf('Train: %d speakers (%d samples)\n', numel(trainSpeakers), sum(trainMask));
fprintf('Test : %d speakers (%d samples)\n', numel(testSpeakers),  sum(testMask));

% Leakage check
if isempty(intersect(unique(speakerIDs(trainMask)), unique(speakerIDs(testMask))))
    fprintf('Speaker leakage check: PASSED\n\n');
else
    warning('Speaker leakage detected!');
end

% =============================================================
% 4.2  Random Forest
% =============================================================
fprintf('Training Random Forest (200 trees)...\n');
nVars   = max(1, round(sqrt(nFeat)));
rfModel = TreeBagger(200, XTrain, yTrain, ...
    'Method',                 'classification', ...
    'OOBPredictorImportance', 'on', ...
    'MinLeafSize',            5, ...
    'NumVariablesToSample',   nVars);

[predRF_cell, scoresRF] = predict(rfModel, XTest);
predRF    = str2double(predRF_cell);
metricsRF = computeMetrics(yTest, predRF, 'Random Forest');
oobAcc    = 1 - oobError(rfModel); oobAcc = oobAcc(end);
fprintf('  OOB accuracy: %.2f%%\n\n', oobAcc*100);

% =============================================================
% 4.3  SVM
% =============================================================
fprintf('Training SVM (RBF kernel)...\n');
svmModel = fitcsvm(XTrain, yTrain, ...
    'KernelFunction','rbf','Standardize',true, ...
    'BoxConstraint',1,'KernelScale','auto');
svmModel = fitPosterior(svmModel);

[predSVM, scoresSVM] = predict(svmModel, XTest);
metricsSVM = computeMetrics(yTest, predSVM, 'SVM (RBF)');

% =============================================================
% 4.4  Leave-One-Speaker-Out cross-validation
% =============================================================
fprintf('\nLeave-One-Speaker-Out cross-validation...\n');
losoAcc = zeros(nSpeakers,1);
for s = 1:nSpeakers
    sp      = uniqueSpeakers{s};
    isTrain = ~strcmp(speakerIDs, sp);
    isTest  =  strcmp(speakerIDs, sp);
    if sum(isTest)==0, continue; end
    mdl = TreeBagger(100, featureMatrix(isTrain,:), labelsClean(isTrain), ...
                     'Method','classification', ...
                     'NumVariablesToSample',nVars,'MinLeafSize',5);
    p   = str2double(predict(mdl, featureMatrix(isTest,:)));
    losoAcc(s) = mean(p == labelsClean(isTest));
end
fprintf('  LOSO CV: %.2f%% +/- %.2f%%\n\n', ...
        mean(losoAcc)*100, std(losoAcc)*100);

% =============================================================
% 4.5  Confusion matrices (custom clear version)
% =============================================================
figure('Name','Confusion Matrices','NumberTitle','off', ...
       'Position',[80 80 1100 480]);

subplot(1,2,1);
plotConfusionMatrix(yTest, predRF, ...
    sprintf('Random Forest  (speaker-stratified, %d test speakers)', nTest), ...
    metricsRF);

subplot(1,2,2);
plotConfusionMatrix(yTest, predSVM, ...
    sprintf('SVM RBF  (speaker-stratified, %d test speakers)', nTest), ...
    metricsSVM);

print(gcf,'-dpng','-r300','confusion_matrices.png');
fprintf('Saved: confusion_matrices.png');

% =============================================================
% 4.6  ROC curves
% =============================================================
classNames   = str2double(rfModel.ClassNames);
fakeCol      = find(classNames==1);
scoreSVM_pos = scoresSVM(:, svmModel.ClassNames==1);

[xRF, yRF,~,aucRF]   = perfcurve(yTest, scoresRF(:,fakeCol), 1);
[xSVM,ySVM,~,aucSVM] = perfcurve(yTest, scoreSVM_pos, 1);

figure('Name','ROC Curves','NumberTitle','off');
hold on;
plot(xRF, yRF, 'b-',  'LineWidth',2,'DisplayName',sprintf('RF  AUC=%.3f',aucRF));
plot(xSVM,ySVM,'r--', 'LineWidth',2,'DisplayName',sprintf('SVM AUC=%.3f',aucSVM));
plot([0 1],[0 1],'k:','HandleVisibility','off');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves – Speaker-Stratified');
legend('Location','southeast'); grid on;
print(gcf,'-dpng','-r300','roc_curves.png');
fprintf('Saved: roc_curves.png\n');

% =============================================================
% 4.7  Feature importance
% =============================================================
imp = rfModel.OOBPermutedPredictorDeltaError;
figure('Name','Feature Importance','NumberTitle','off');
bar(imp); title('RF Feature Importance');
xlabel('Feature index (1-88)'); ylabel('Importance'); grid on;
print(gcf,'-dpng','-r300','rf_feature_importance.png');
fprintf('Saved: rf_feature_importance.png\n');

% =============================================================
% 4.8  Per-speaker accuracy chart
% =============================================================
figure('Name','Per-Speaker Accuracy','NumberTitle','off', ...
       'Position',[100 100 900 420]);
bar(losoAcc*100,'FaceColor',[0.2 0.5 0.8]);
yline(mean(losoAcc)*100,'r--','LineWidth',1.5, ...
      'Label',sprintf('Mean %.1f%%',mean(losoAcc)*100));
set(gca,'XTick',1:nSpeakers,'XTickLabel',uniqueSpeakers, ...
        'XTickLabelRotation',45,'FontSize',9);
ylabel('Accuracy (%)'); title('Per-Speaker LOSO Accuracy'); grid on;
print(gcf,'-dpng','-r300','per_speaker_accuracy.png');
fprintf('Saved: per_speaker_accuracy.png\n');

% =============================================================
% 4.9  Summary
% =============================================================
fprintf('\n=== Results Summary ===\n');
fprintf('%-16s | Acc    | Prec   | Recall | F1     | AUC\n','Method');
fprintf('%-16s +--------+--------+--------+--------+------\n','----------------');
printRow('Random Forest', metricsRF,  aucRF);
printRow('SVM (RBF)',     metricsSVM, aucSVM);
fprintf('\nLOSO CV (RF): %.2f%% +/- %.2f%%\n', mean(losoAcc)*100, std(losoAcc)*100);
fprintf('OOB Acc  (RF): %.2f%%\n\n', oobAcc*100);

save('best_rf_model.mat', 'rfModel', 'nFeat');
save('best_svm_model.mat', 'svmModel');
disp('Models saved. Run deepfake_detector_Live.m next.');

% =============================================================
% Helpers
% =============================================================
function m = computeMetrics(yTrue, yPred, name)
    TP=sum(yPred==1&yTrue==1); FP=sum(yPred==1&yTrue==0);
    FN=sum(yPred==0&yTrue==1);
    m.Accuracy  = mean(yPred==yTrue);
    m.Precision = TP/max(TP+FP,1);
    m.Recall    = TP/max(TP+FN,1);
    m.F1        = 2*m.Precision*m.Recall/max(m.Precision+m.Recall,eps);
    fprintf('  %s  ->  Acc=%.2f%%  Prec=%.2f%%  Recall=%.2f%%  F1=%.2f%%\n', ...
            name,m.Accuracy*100,m.Precision*100,m.Recall*100,m.F1*100);
end

function printRow(name, m, auc)
    fprintf('%-16s | %5.2f%% | %6.2f%% | %6.2f%% | %6.2f%% | %.3f\n', ...
            name,m.Accuracy*100,m.Precision*100,m.Recall*100,m.F1*100,auc);
end

function plotConfusionMatrix(yTrue, yPred, titleStr, metrics)
    % Compute confusion matrix values
    TP = sum(yPred==1 & yTrue==1);
    TN = sum(yPred==0 & yTrue==0);
    FP = sum(yPred==1 & yTrue==0);
    FN = sum(yPred==0 & yTrue==1);

    CM = [TN FP; FN TP];   % rows=actual, cols=predicted, [Real;Fake] x [Real,Fake]

    % Colour map: light for low, dark blue for high
    imagesc(CM);
    colormap(gca, [0.97 0.97 0.97; 0.85 0.92 0.98; 0.55 0.75 0.92; 0.18 0.49 0.80]);
    caxis([0 max(CM(:))+1]);

    % Cell labels — count + percentage
    labels = {'Real','Fake'};
    for r = 1:2
        for c = 1:2
            val = CM(r,c);
            pct = 100 * val / max(sum(CM(r,:)),1);
            if val > max(CM(:))*0.5
                txtcol = [1 1 1];
            else
                txtcol = [0.1 0.1 0.1];
            end
            text(c, r, sprintf('%d\n%.1f%%', val, pct), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontSize', 15, ...
                'FontWeight','bold', ...
                'Color', txtcol);
        end
    end

    % Axes formatting
    set(gca, 'XTick',[1 2], 'XTickLabel',labels, ...
             'YTick',[1 2], 'YTickLabel',labels, ...
             'FontSize', 13, 'TickLength',[0 0], ...
             'XAxisLocation','bottom');
    xlabel('Predicted Label', 'FontSize',13, 'FontWeight','bold');
    ylabel('Actual Label',    'FontSize',13, 'FontWeight','bold');
    title(titleStr, 'FontSize', 12);

    % Metrics box below title
    annotation_str = sprintf('Acc: %.1f%%   Prec: %.1f%%   Recall: %.1f%%   F1: %.1f%%', ...
        metrics.Accuracy*100, metrics.Precision*100, ...
        metrics.Recall*100,   metrics.F1*100);
    xlabel({annotation_str, '', 'Predicted Label'}, 'FontSize', 11);

    % Diagonal highlight border
    hold on;
    rectangle('Position',[0.5 0.5 1 1],'EdgeColor',[0.1 0.6 0.2],'LineWidth',2.5);
    rectangle('Position',[1.5 1.5 1 1],'EdgeColor',[0.1 0.6 0.2],'LineWidth',2.5);
    hold off;
end
