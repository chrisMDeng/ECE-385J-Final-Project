% Setup EEG Data Arrays
%[match_yes1, match_no1, nomatch_yes1, nomatch_no1] = setup_data(1);
%[match_yes, match_no, nomatch_yes, nomatch_no] = deal(match_yes1, match_no1, nomatch_yes1, nomatch_no1);

[match_yes2, match_no2, nomatch_yes2, nomatch_no2] = setup_data(2);
[match_yes, match_no, nomatch_yes, nomatch_no] = deal(match_yes2, match_no2, nomatch_yes2, nomatch_no2);

%[match_yes3, match_no3, nomatch_yes3, nomatch_no3] = setup_data(3);
%[match_yes, match_no, nomatch_yes, nomatch_no] = deal(match_yes3, match_no3, nomatch_yes3, nomatch_no3);

%match_yes = [match_yes1, match_yes2, match_yes3];
%match_no = [match_no1, match_no2, match_no3];
%nomatch_yes = [nomatch_yes1, nomatch_yes2, nomatch_yes3];
%nomatch_no = [nomatch_no1, nomatch_no2, nomatch_no3];
%%
a = [match_yes, match_no, nomatch_yes, nomatch_no];

size(a)

lengths = cellfun(@(x) size(x, 2), a);

figure;
histogram(lengths, 'BinWidth', 10);   % choose bin size you want
xlabel('Length Range (samples)');
ylabel('Number of Cells');
title('Distribution of Cell Lengths');
grid on;


%%
Fs = 512;
end_index = 250;
C = [0 1; 1 0];        % rows = true class, cols = predicted class

p300_epochs    = get_rows([match_yes, nomatch_yes], end_index);    % positive class
nonp300_epochs = get_rows([nomatch_no1, nomatch_yes], end_index);   % negative class



epochs = [p300_epochs, nonp300_epochs];  % 1 x N cell array
labels = [ones(1, numel(p300_epochs)), zeros(1, numel(nonp300_epochs))];  % 1 x N
labels = labels(:);   % N x 1
N = numel(epochs);

%fprintf('%d  %d  %d  %d  %d\n\n', ...
%    size(a, 2), size(match_yes,2), size(match_no,2), size(nomatch_yes,2), size(nomatch_no,2));

%fprintf('%d  %d  %d\n\n', ...
%    size(epochs, 2), size(p300_epochs, 2), size(nonp300_epochs, 2));

%-----------------------------
pz_idx = 26;        % which channel to extract (if needed)
N = numel(epochs);  % number of trials

% Round everything to integer sample indices
baseline_start = 1;
baseline_end   = 52;

p300_start     = 155;
p300_end       = end_index;

%-----------------
nFeat = 4;                   
X = zeros(N, nFeat);        

for k = 1:N
    
    ep = epochs{k};         % nChan x nSamples
    sig = ep;               % if only one channel; otherwise sig = ep(pz_idx,:)
    L = length(sig);

    % --------------------
    % 1. CLIP WINDOWS TO SIGNAL LENGTH
    % --------------------
    bs = max(1, baseline_start);
    be = min(L, baseline_end);

    ps = max(1, p300_start);
    pe = min(L, p300_end);

    % --------------------
    % 2. BASELINE CORRECTION
    % --------------------
    base_mean = mean(sig(bs:be));
    sig_bc = sig - base_mean;

    % --------------------
    % 3. Extract P300 window (only if long enough)
    % --------------------
    if pe > ps   % valid window
        p300_seg = sig_bc(ps:pe);
    else
        p300_seg = NaN;    % or continue; or zeros
    end

    % --------------------
    % 4. Feature extraction
    % --------------------
    f1 = mean(p300_seg);
    f2 = max(p300_seg);
    f3 = min(p300_seg);
    f4 = trapz(p300_seg);

    X(k,:) = [f1, f2, f3, f4];
end

%---------------
pos_frac = mean(labels == 1);    % ≈ 0.08
neg_frac = mean(labels == 0);    % ≈ 0.92

%{
w_pos = 1 / pos_frac;
w_neg = 1 / neg_frac;

C = [0      w_pos;      % cost(pred=0 | true=1)
     w_neg  0   ];      % cost(pred=1 | true=0)
%}



%---------------
cv = cvpartition(labels, 'HoldOut', 0.3);
Xtrain = X(training(cv), :);
ytrain = labels(training(cv));
Xtest  = X(test(cv), :);
ytest  = labels(test(cv));

SVMModel = fitcsvm(Xtrain, ytrain, ...
    'KernelFunction','rbf', ...
    'Standardize',true, ...
    'ClassNames',[0 1], ...
    'Cost', C);

ypred = predict(SVMModel, Xtest);
acc   = mean(ypred == ytest);
fprintf('Test accuracy = %.3f  --> %d\n', acc, end_index);

confusionchart(ytest, ypred);
title("Confusion Matrix");
