% =============================================================
% extractSpeakerIDs.m
% Called automatically by Classifier_4.m — do NOT run manually.
%
% Your naming pattern:
%   Real: "Wasif script 1.wav"           -> wasif
%   Fake: "Wasif 1 (Fake).wav"           -> wasif
%   Real: "Borhanul sami script 1 .wav"  -> borhanul sami
%   Fake: "Borhanul sami 1 (Fake).wav"   -> borhanul sami
% =============================================================

function ids = extractSpeakerIDs(filePaths)
    n   = numel(filePaths);
    ids = cell(n,1);

    for i = 1:n
        [~, fname] = fileparts(filePaths{i});

        % 1. Lowercase everything first
        s = lower(fname);

        % 2. Strip (fake) with any surrounding punctuation/spaces
        s = regexprep(s, '\s*[\-_]?\s*\(fake\)\s*', ' ');

        % 3. Strip standalone word "fake" at end
        s = regexprep(s, '[\s\-_]+fake\s*$', '');

        % 4. Strip the word "script" from ANYWHERE in the string
        s = regexprep(s, '\s*script\s*', ' ');

        % 5. Strip trailing numbers including formats like 2(1), (1), 1
        s = regexprep(s, '\s*\(\d+\)\s*$', '');  % remove (1) at end
        s = regexprep(s, '[\s\-_]+\d+\s*$', ''); % remove trailing number

        % 6. Replace dashes and underscores with spaces
        s = regexprep(s, '[\-_]+', ' ');

        % 7. Collapse multiple spaces and trim
        s = strtrim(regexprep(s, '\s+', ' '));

        % 8. Fix known typos and inconsistencies
        s = fixName(s);

        if isempty(s)
            s = sprintf('speaker%03d', i);
        end

        ids{i} = s;
    end

    printPreview(filePaths, ids);
end

% =============================================================
% Fix known name inconsistencies in your dataset
% =============================================================
function name = fixName(name)
    fixes = {
    % ===== Existing =====
    'swakhorr',          'swakhor';
    'twaha',             'tawha';
    'borhan sami',       'borhanul sami';
    'name1',             'name 1';
    'sample2',           'sample 2';
    'sample3',           'sample 3';
    'himel',             'himel';
    'iftakhar',          'iftakhar';
    'jishan',            'jishan';
    'jubayer',           'jubayer';
    'mahfuj',            'mahfuj';
    'mehadi',            'mehadi';
    'miraz',             'miraz';
    'mozahid',           'mozahid';
    'mubid',             'mubid';
    'mursalin',          'mursalin';
    'nazmus sakib',      'nazmus sakib';
    'niloy r roommate',  'niloy roommate';
    'person 2',          'person 2';
    'ratul',             'ratul';

    % ===== NEW FIXES =====

    % --- Remove "one/two/three" variants for ID speakers ---
    '2118001 one',   '2118001';
    '2118001 two',   '2118001';
    '2118001 three', '2118001';

    '2118013 one',   '2118013';
    '2118013 two',   '2118013';
    '2118013 three', '2118013';

    '2118014 one',   '2118014';
    '2118014 two',   '2118014';
    '2118014 three', '2118014';

    '2118018 one',   '2118018';
    '2118018 two',   '2118018';
    '2118018 three', '2118018';

    '2118028 one',   '2118028';
    '2118028 two',   '2118028';
    '2118028 three', '2118028';

    '2118037 one',   '2118037';
    '2118037 two',   '2118037';
    '2118037 three', '2118037';

    '2118039 one',   '2118039';
    '2118039 two',   '2118039';
    '2118039 three', '2118039';

    % --- Fix script typos ---
    'achal scipt1',  'achal';
    'mamo scipt3',   'mamo';

    % --- Fix spelling inconsistencies ---
    'aidba',         'adiba';
    'such i',        'suchi';
    };
    for k = 1:size(fixes,1)
        if strcmp(name, fixes{k,1})
            name = fixes{k,2};
            return;
        end
    end
end

% =============================================================
function printPreview(filePaths, ids)
    fprintf('\n=== Speaker ID extraction preview (first 20) ===\n');
    fprintf('%-45s  ->  %s\n', 'Filename', 'Speaker ID');
    fprintf('%s\n', repmat('-',1,65));

    nShow = min(20, numel(filePaths));
    for i = 1:nShow
        [~,fn,ex] = fileparts(filePaths{i});
        d = [fn ex];
        if length(d)>44, d=['...' d(end-40:end)]; end
        fprintf('%-45s  ->  %s\n', d, ids{i});
    end
    if numel(filePaths)>20
        fprintf('  ... and %d more\n', numel(filePaths)-20);
    end

    uids = unique(ids);
    fprintf('\nTotal files: %d   Unique speakers: %d\n', numel(filePaths), numel(uids));
    fprintf('Speakers: %s\n\n', strjoin(sort(uids), ', '));

    anyWarn = false;
    for k = 1:numel(uids)
        cnt = sum(strcmp(ids, uids{k}));
        if cnt ~= 6
            if ~anyWarn
                fprintf('File count warnings (expected 6 per speaker):\n');
                anyWarn = true;
            end
            fprintf('  "%s" has %d files\n', uids{k}, cnt);
        end
    end
    if ~anyWarn
        fprintf('All speakers have exactly 6 files.\n');
    end
    fprintf('\n');
end
