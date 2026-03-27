% =============================================================
% Step 3 – EDA: Real vs Fake Comparison
% Produces 5 figures saved as PNG:
%   eda_01_mfcc_boxplot.png
%   eda_02_correlation_bar.png
%   eda_03_histograms.png
%   eda_04_pca_scatter.png
%   eda_05_spectrograms.png
%
% Requires: feature_matrix_final_safe.mat
%           labels_clean_final.mat
%           dataset_table.mat
% =============================================================

clc; close all;

% Load data
if ~exist('featureMatrix','var') || isempty(featureMatrix)
    if isfile('feature_matrix_final_safe.mat')
        load('feature_matrix_final_safe.mat', 'featureMatrix');
        load('labels_clean_final.mat',         'labelsClean');
        fprintf('Loaded feature matrix.\n');
    else
        error('feature_matrix_final_safe.mat not found. Run Feature_Extraction.m first.');
    end
end
if ~exist('T','var') && isfile('dataset_table.mat')
    load('dataset_table.mat', 'T');
end

realIdx = labelsClean == 0;
fakeIdx = labelsClean == 1;
nReal   = sum(realIdx);
nFake   = sum(fakeIdx);
nFeat   = size(featureMatrix, 2);

fprintf('Real: %d   Fake: %d   Features: %d\n\n', nReal, nFake, nFeat);

featNames = buildFeatureNames(nFeat);

% =============================================================
% Figure 1 – MFCC Mean Boxplot
% =============================================================
figure('Name','MFCC Means','NumberTitle','off','Position',[50 50 1300 550]);
boxplot(featureMatrix(:,1:13), ...
    'Labels', arrayfun(@(k) sprintf('C%d',k),1:13,'UniformOutput',false), ...
    'Symbol','r+','Colors',[0.75 0.75 0.75],'Whisker',1.5);
hold on;
hR = scatter(repmat(1:13,nReal,1), featureMatrix(realIdx,1:13), ...
             35, 'b', 'filled', 'MarkerFaceAlpha',0.55);
hF = scatter(repmat(1:13,nFake,1), featureMatrix(fakeIdx,1:13), ...
             35, 'r', 'filled', 'MarkerFaceAlpha',0.55);
legend([hR(1) hF(1)],{'Real','Fake'},'Location','bestoutside','FontSize',11);
title('MFCC Mean Coefficients – Real vs Fake','FontSize',14);
xlabel('MFCC Coefficient'); ylabel('Mean Value'); grid on; hold off;
saveFig(gcf,'eda_01_mfcc_boxplot.png');

% =============================================================
% Figure 2 – Pearson Correlation Bar
% =============================================================
classVec = double(labelsClean);
corrVals = zeros(1,nFeat);
for col = 1:nFeat
    v = featureMatrix(:,col);
    valid = ~isnan(v);
    if sum(valid) > 5
        r = corrcoef(v(valid), classVec(valid));
        corrVals(col) = r(1,2);
    end
end
[sortedCorr, sortIdx] = sort(abs(corrVals),'descend');
top20 = sortIdx(1:min(20,nFeat));

figure('Name','Feature Correlation','NumberTitle','off','Position',[50 50 900 600]);
bh = barh(sortedCorr(1:min(20,nFeat)),'FaceColor','flat');
for k = 1:length(top20)
    bh.CData(k,:) = [0.2 0.45 0.8];
end
set(gca,'YTick',1:min(20,nFeat),'YTickLabel',featNames(top20), ...
        'YDir','reverse','FontSize',10);
xlabel('|Pearson r| with class label');
title('Top-20 Features by Correlation with Class','FontSize',13); grid on;
saveFig(gcf,'eda_02_correlation_bar.png');

% =============================================================
% Figure 3 – Histogram Overlays (top 6 features)
% =============================================================
figure('Name','Feature Distributions','NumberTitle','off','Position',[50 50 1300 650]);
for k = 1:6
    subplot(2,3,k);
    col = top20(k);
    rv  = featureMatrix(realIdx,col); rv = rv(~isnan(rv));
    fv  = featureMatrix(fakeIdx,col); fv = fv(~isnan(fv));
    edges = linspace(min([rv;fv]), max([rv;fv]), 35);
    histogram(rv,edges,'Normalization','probability', ...
              'FaceColor','b','FaceAlpha',0.5,'EdgeColor','none');
    hold on;
    histogram(fv,edges,'Normalization','probability', ...
              'FaceColor','r','FaceAlpha',0.5,'EdgeColor','none');
    hold off;
    title(featNames{col},'FontSize',10,'Interpreter','none');
    xlabel('Value'); ylabel('Prob.');
    legend({'Real','Fake'},'Location','best','FontSize',8); grid on;
end
sgtitle('Distribution – Top 6 Discriminative Features','FontSize',13);
saveFig(gcf,'eda_03_histograms.png');

% =============================================================
% Figure 4 – PCA 2D Scatter
% =============================================================
validMask = ~any(isnan(featureMatrix),2);
Xv  = featureMatrix(validMask,:);
lv  = labelsClean(validMask);
mu  = mean(Xv); sg = std(Xv); sg(sg==0)=1;
[~,score,~,~,explained] = pca((Xv-mu)./sg);

figure('Name','PCA Scatter','NumberTitle','off','Position',[50 50 750 600]);
scatter(score(lv==0,1),score(lv==0,2),30,'b','filled', ...
        'MarkerFaceAlpha',0.5,'DisplayName','Real'); hold on;
scatter(score(lv==1,1),score(lv==1,2),30,'r','filled', ...
        'MarkerFaceAlpha',0.5,'DisplayName','Fake'); hold off;
xlabel(sprintf('PC1 (%.1f%% var)',explained(1)));
ylabel(sprintf('PC2 (%.1f%% var)',explained(2)));
title('PCA – Real vs Fake Feature Space','FontSize',13);
legend('Location','best'); grid on;
saveFig(gcf,'eda_04_pca_scatter.png');

% =============================================================
% Figure 5 – Spectrogram Comparison
% =============================================================
if exist('T','var')
    exReal = pickFile(T.FilePath(T.Label==0));
    exFake = pickFile(T.FilePath(T.Label==1));
    if ~isempty(exReal) && ~isempty(exFake)
        figure('Name','Spectrograms','NumberTitle','off','Position',[50 50 1200 450]);
        subplot(1,2,1); plotSpec(exReal,'Real Voice');
        subplot(1,2,2); plotSpec(exFake,'Fake (AI) Voice');
        sgtitle('Spectrogram Comparison – Real vs Fake','FontSize',13);
        saveFig(gcf,'eda_05_spectrograms.png');
    end
end

% t-test summary
fprintf('=== t-test (first 26 features) ===\n');
pValues = nan(1,nFeat);
for col = 1:nFeat
    rv = featureMatrix(realIdx,col); rv=rv(~isnan(rv));
    fv = featureMatrix(fakeIdx,col); fv=fv(~isnan(fv));
    if numel(rv)<5||numel(fv)<5, continue; end
    [~,pValues(col)] = ttest2(rv,fv);
    if col<=26
        fprintf('  %-22s  p=%.2e  %s\n',featNames{col},pValues(col),pSig(pValues(col)));
    end
end
fprintf('\n%d/%d features significant at p<0.05\n', sum(pValues<0.05,'omitnan'),nFeat);
fprintf('\nAll figures saved. Next: run Classifier.m\n');

% =============================================================
% Helpers
% =============================================================
function names = buildFeatureNames(n)
    names = cell(1,n);
    groups = {'MFCC mean','MFCC std','Delta mean','Delta std', ...
              'DeltaDelta mean','DeltaDelta std'};
    k=1;
    for g=1:6
        for c=1:13
            if k>n, break; end
            names{k}=sprintf('%s %d',groups{g},c); k=k+1;
        end
    end
    spec={'ZCR mean','ZCR std','RMS mean','RMS std','SC mean', ...
          'SC std','SB mean','SB std','SR mean','SR std'};
    for s=1:length(spec)
        if k>n, break; end
        names{k}=spec{s}; k=k+1;
    end
    for i=k:n, names{i}=sprintf('Feat %d',i); end
end

function s = pSig(p)
    if p<0.001, s='*** highly sig.';
    elseif p<0.01, s='**';
    elseif p<0.05, s='*';
    else, s=''; end
end

function plotSpec(fp, ttl)
    try
        [y,fs]=audioread(fp);
        if size(y,2)>1,y=mean(y,2); end
        if fs~=16000,y=resample(y,16000,fs); end
        spectrogram(y,hamming(512),256,512,16000,'yaxis','MinThreshold',-90);
        title(ttl); colorbar;
    catch
        title([ttl ' (error)']);
    end
end

function fp = pickFile(list)
    fp='';
    for i=1:numel(list)
        if isfile(list{i}), fp=list{i}; return; end
    end
end

function saveFig(fig, fname)
    print(fig,'-dpng','-r300',fname);
    fprintf('  Saved: %s\n',fname);
end
