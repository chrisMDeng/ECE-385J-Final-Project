% Setup EEG Data Arrays
[match_yes1, match_no1, match_idk1, nomatch_yes1, nomatch_no1, nomatch_idk1] = setup_data(1);
[match_yes, match_no, match_idk, nomatch_yes, nomatch_no, nomatch_idk] = deal(match_yes1, match_no1, match_idk1, nomatch_yes1, nomatch_no1, nomatch_idk1);

%[match_yes2, match_no2, match_idk2, nomatch_yes2, nomatch_no2, nomatch_idk2] = setup_data(2);
%[match_yes, match_no, match_idk, nomatch_yes, nomatch_no, nomatch_idk] = deal(match_yes2, match_no2, match_idk2, nomatch_yes2, nomatch_no2, nomatch_idk2);

%[match_yes3, match_no3, match_idk3, nomatch_yes3, nomatch_no3, nomatch_idk3] = setup_data(3);
%[match_yes, match_no, match_idk, nomatch_yes, nomatch_no, nomatch_idk] = deal(match_yes3, match_no3, match_idk3, nomatch_yes3, nomatch_no3, nomatch_idk3);


match_yes = [match_yes1, match_yes2, match_yes3];
match_no = [match_no1, match_no2, match_no3];
match_idk = [match_idk1, match_idk2, match_idk3];
nomatch_yes = [nomatch_yes1, nomatch_yes2, nomatch_yes3];
nomatch_no = [nomatch_no1, nomatch_no2, nomatch_no3];
nomatch_idk = [nomatch_idk1, nomatch_idk2, nomatch_idk3];


%% Plot a single graph
a = [match_yes, match_no, match_idk, nomatch_yes, nomatch_no, nomatch_idk];

%a = match_yes;

size(a);

lengths = cellfun(@(x) size(x, 2), a);

figure;
histogram(lengths, 'BinWidth', 10);   % choose bin size you want
xlabel('Length Range (samples)');
ylabel('Number of Cells');
title('Distribution of Cell Lengths');
grid on;



%% Set up arrays
Fs = 512;
end_index = 180;        % keeps only trials where the samples are >= end_index

p300_epochs    = get_rows([match_yes, nomatch_no], end_index);    % positive class
nonp300_epochs = get_rows([match_no, nomatch_yes], end_index);   % negative class
%p300_epochs    = get_rows([match_yes], end_index);    % positive class
%nonp300_epochs = get_rows([nomatch_yes], end_index);   % negative class

epochs = [p300_epochs, nonp300_epochs];  % 1 x N cell array



%% Temporal Filtering (REGULAR)
%60Hz Notch Filter
f0 = 60;                          % Hz
wo = f0 / (Fs/2);                 % normalized frequency
Q  = 35;                          % quality factor (narrow notch)
bw = wo / Q;

[b_notch, a_notch] = iirnotch(wo, bw);   % causal IIR notch

% Bandpass Filter
bp_low  = 3;                    % Hz
bp_high = 10;                     % Hz

Wn = [bp_low bp_high] / (Fs/2);

[b_bp, a_bp] = butter(4, Wn, 'bandpass');    % causal Butterworth

% Apply filters to epochs
Ntr = numel(epochs);
epochs_filt = cell(size(epochs));

for k = 1:Ntr

    % extract 1 x 1 x n_k and convert to n_k x 1
    sig = squeeze(epochs{k});
    sig = sig(:);   % ensure column

    % --- Causal 60 Hz notch ---
    sig = filter(b_notch, a_notch, sig);

    % --- Causal P300 bandpass ---
    sig = filter(b_bp, a_bp, sig);

    % store back in 1 x 1 x n_k format
    epochs_filt{k} = reshape(sig, [1 numel(sig)]);
end


%% Temporal Filtering (split between short and long)

Fs = 512;   % your sample rate
split_length = 310;   % threshold for short vs long

% -----------------------------
% 60 Hz Notch Filter (shared)
% -----------------------------
f0 = 60;                          % Hz
wo = f0 / (Fs/2);                 % normalized frequency
Q  = 35;                          % quality factor (narrow notch)
bw = wo / Q;

[b_notch, a_notch] = iirnotch(wo, bw);   % causal IIR notch

% -----------------------------
% Bandpass Filter 1 (for SHORT epochs)
%   example: 0.1–10 Hz
% -----------------------------
bp1_low  = 0.1;                   % Hz
bp1_high = 10;                    % Hz

Wn1 = [bp1_low bp1_high] / (Fs/2);
[b_bp_short, a_bp_short] = butter(4, Wn1, 'bandpass');   % causal

% -----------------------------
% Bandpass Filter 2 (for LONG epochs)
%   example: 0.1–15 Hz (change as you like)
% -----------------------------
bp2_low  = 3;                   % Hz
bp2_high = 10;                    % Hz

Wn2 = [bp2_low bp2_high] / (Fs/2);
[b_bp_long, a_bp_long] = butter(4, Wn2, 'bandpass');    % causal

% -----------------------------
% Apply filters to epochs
% epochs{k} is 1 x 1 x n_k
% -----------------------------
Ntr = numel(epochs);
epochs_filt = cell(size(epochs));
lengths = zeros(Ntr,1);    % keep lengths if you want to reuse later

for k = 1:Ntr

    % extract 1 x 1 x n_k and convert to n_k x 1
    sig = squeeze(epochs{k});
    sig = sig(:);   % column
    L = numel(sig);
    lengths(k) = L;

    % --- Shared 60 Hz notch ---
    sig = filter(b_notch, a_notch, sig);

    % --- Length-dependent bandpass ---
    if L < split_length
        % SHORT epoch -> use bandpass 1
        sig = filter(b_bp_short, a_bp_short, sig);
    else
        % LONG epoch -> use bandpass 2
        sig = filter(b_bp_long, a_bp_long, sig);
    end

    % store back; use 1xn to match earlier code that uses squeeze()
    epochs_filt{k} = reshape(sig, [1 numel(sig)]);
end


%% Show Average Lines

% Split arrays a and b
a = epochs_filt(1:numel(p300_epochs));
b = epochs_filt(numel(p300_epochs)+1:end);

% Helper for 1x1xn arrays
get_len = @(x) numel(x);        % length of epoch

% --- Process A ---
len_a = cellfun(get_len, a);
short_a = a(len_a < 310 & len_a > 50);
long_a  = a(len_a >= 310);

avg_short_a = avg_row(short_a);
avg_long_a  = avg_row(long_a);

% --- Process B ---
len_b = cellfun(get_len, b);
short_b = b(len_b < 310 & len_b > 50);
long_b  = b(len_b >= 310);

avg_short_b = avg_row(short_b);
avg_long_b  = avg_row(long_b);

% Time vectors
Fs = 512;

t_short_a = (0:numel(avg_short_a)-1)/Fs;
t_long_a  = (0:numel(avg_long_a)-1)/Fs;

t_short_b = (0:numel(avg_short_b)-1)/Fs;
t_long_b  = (0:numel(avg_long_b)-1)/Fs;

%   FIGURE 1: SHORT (A-short + B-short)
figure;
hold on;
plot(t_short_a, avg_short_a, 'LineWidth', 2);
plot(t_short_b, avg_short_b, 'LineWidth', 2);
xlabel("Time (s)");
ylabel("Amplitude");
title("SHORT Epochs (A-short and B-short)");
legend("A Short", "B Short");
grid on;
hold off;

%   FIGURE 2: LONG (A-long + B-long)
figure;
hold on;
plot(t_long_a, avg_long_a, 'LineWidth', 2);
plot(t_long_b, avg_long_b, 'LineWidth', 2);
xlabel("Time (s)");
ylabel("Amplitude");
title("LONG Epochs (A-long and B-long)");
legend("A Long", "B Long");
grid on;
hold off;



%% SVM Regular

C = [0 1; 3.43 0];        % rows = true class, cols = predicted class

labels = [ones(1, numel(p300_epochs)), zeros(1, numel(nonp300_epochs))];  % 1 x N
labels = labels(:);   % N x 1

%fprintf('%d  %d  %d  %d  %d\n\n', ...
%    size(a, 2), size(match_yes,2), size(match_no,2), size(nomatch_yes,2), size(nomatch_no,2));

%fprintf('%d  %d  %d\n\n', ...
%    size(epochs, 2), size(p300_epochs, 2), size(nonp300_epochs, 2));

%-----------------------------
pz_idx = 26;        % which channel to extract (if needed)
N = numel(epochs_filt);  % number of trials

% Round everything to integer sample indices
baseline_start = 1;
%baseline_end   = 52;
baseline_end   = 10;

%p300_start     = 155;
p300_start     = 20;
p300_end       = end_index;

%-----------------
nFeat = 4;                   
X = zeros(N, nFeat);        

for k = 1:N
    
    ep = epochs_filt{k};         % nChan x nSamples
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
fprintf('Test accuracy = %.3f  --> %d, %d\n', acc, end_index, C(2, 1));

confusionchart(ytest, ypred);
title("Confusion Matrix");


%% Split setup

%  Setup and labels

% Two separate cost matrices

labels = [ones(1, numel(p300_epochs)), zeros(1, numel(nonp300_epochs))];  % 1 x N
labels = labels(:);   % N x 1

pz_idx = 26;
N = numel(epochs_filt);

baseline_start = 1;
baseline_end   = 10;

p300_start = 20;
p300_end   = end_index;

nFeat = 4;
X = zeros(N, nFeat);
lengths = zeros(N, 1);

%  Feature extraction loop
for k = 1:N

    ep = epochs_filt{k};
    sig = squeeze(ep);
    sig = sig(:).';
    L = length(sig);
    lengths(k) = L;

    bs = max(1, baseline_start);
    be = min(L, baseline_end);

    ps = max(1, p300_start);
    pe = min(L, p300_end);

    base_mean = mean(sig(bs:be));
    sig_bc = sig - base_mean;

    if pe > ps
        p300_seg = sig_bc(ps:pe);
    else
        p300_seg = zeros(1, max(pe-ps+1,1));
    end

    f1 = mean(p300_seg);
    f2 = max(p300_seg);
    f3 = min(p300_seg);
    f4 = trapz(p300_seg);

    X(k,:) = [f1, f2, f3, f4];
end

%  Split into short vs long sets _____________________ SPLIT LENGTH
split_length = 310;

idx_short = lengths <  split_length;
idx_long  = lengths >= split_length;

X_short = X(idx_short, :);
y_short = labels(idx_short);

X_long  = X(idx_long, :);
y_long  = labels(idx_long);

fprintf('Short trials: %d   Long trials: %d\n', sum(idx_short), sum(idx_long));

%% SVM Model

C_short = [0 1;   4.62 0];     % (example)
C_long  = [0 1;   2.56 0];     % (example)

%  Train SHORT model
cv_short = cvpartition(y_short, 'HoldOut', 0.3);

Xtrain_s = X_short(training(cv_short), :);
ytrain_s = y_short(training(cv_short));
Xtest_s  = X_short(test(cv_short), :);
ytest_s  = y_short(test(cv_short));

SVM_short = fitcsvm(Xtrain_s, ytrain_s, ...
    'KernelFunction','rbf', ...
    'Standardize',true, ...
    'ClassNames',[0 1], ...
    'Cost', C_short);

ypred_s = predict(SVM_short, Xtest_s);
acc_s   = mean(ypred_s == ytest_s);
fprintf('SHORT model accuracy = %.3f\n', acc_s);

figure;
confusionchart(ytest_s, ypred_s);
title("Confusion Matrix - SHORT (< 310)");

%  Train LONG model
cv_long = cvpartition(y_long, 'HoldOut', 0.3);

Xtrain_l = X_long(training(cv_long), :);
ytrain_l = y_long(training(cv_long));
Xtest_l  = X_long(test(cv_long), :);
ytest_l  = y_long(test(cv_long));

SVM_long = fitcsvm(Xtrain_l, ytrain_l, ...
    'KernelFunction','rbf', ...
    'Standardize',true, ...
    'ClassNames',[0 1], ...
    'Cost', C_long);

ypred_l = predict(SVM_long, Xtest_l);
acc_l   = mean(ypred_l == ytest_l);
fprintf('LONG model accuracy  = %.3f  --> end_index=%d\n', acc_l, end_index);

figure;
confusionchart(ytest_l, ypred_l);
title("Confusion Matrix - LONG (>= 310)");

%  Overall confusion matrix
ytest_all = [ytest_s; ytest_l];
ypred_all = [ypred_s; ypred_l];

figure;
confusionchart(ytest_all, ypred_all);
title("Confusion Matrix - OVERALL (Short + Long)");

%% LDA Model

C_short = [0 1;   5.5 0];     % (example)
C_long  = [0 1;   2.9 0];     % (example)

%  Train SHORT model (LDA instead of SVM)
cv_short = cvpartition(y_short, 'HoldOut', 0.3);

Xtrain_s = X_short(training(cv_short), :);
ytrain_s = y_short(training(cv_short));
Xtest_s  = X_short(test(cv_short), :);
ytest_s  = y_short(test(cv_short));

LDA_short = fitcdiscr(Xtrain_s, ytrain_s, ...
    'DiscrimType','linear', ...     % LDA
    'ClassNames',[0 1], ...
    'Cost', C_short);               % your cost matrix

ypred_s = predict(LDA_short, Xtest_s);
acc_s   = mean(ypred_s == ytest_s);
fprintf('SHORT LDA model accuracy = %.3f\n', acc_s);

figure;
confusionchart(ytest_s, ypred_s);
title("Confusion Matrix - SHORT (< 310)");


%  Train LONG model (LDA instead of SVM)
cv_long = cvpartition(y_long, 'HoldOut', 0.3);

Xtrain_l = X_long(training(cv_long), :);
ytrain_l = y_long(training(cv_long));
Xtest_l  = X_long(test(cv_long), :);
ytest_l  = y_long(test(cv_long));

LDA_long = fitcdiscr(Xtrain_l, ytrain_l, ...
    'DiscrimType','linear', ...     % LDA
    'ClassNames',[0 1], ...
    'Cost', C_long);                % your cost matrix

ypred_l = predict(LDA_long, Xtest_l);
acc_l   = mean(ypred_l == ytest_l);
fprintf('LONG LDA model accuracy  = %.3f  --> end_index=%d\n', acc_l, end_index);

figure;
confusionchart(ytest_l, ypred_l);
title("Confusion Matrix - LONG (>= 310)");


%  Overall confusion matrix (SHORT + LONG)
ytest_all = [ytest_s; ytest_l];
ypred_all = [ypred_s; ypred_l];

figure;
confusionchart(ytest_all, ypred_all);
title("Confusion Matrix - OVERALL (Short + Long)");


%% Regular LDA

% Cost matrix (unchanged)
C = [0 1; 3.8 0];        % rows = true class, cols = predicted class

labels = [ones(1, numel(p300_epochs)), zeros(1, numel(nonp300_epochs))];  % 1 x N
labels = labels(:);   % N x 1

%-----------------------------
pz_idx = 26;        % which channel to extract (if needed)
N = numel(epochs_filt);  % number of trials

baseline_start = 1;
baseline_end   = 10;

p300_start     = 20;
p300_end       = end_index;

%-----------------
nFeat = 4;                   
X = zeros(N, nFeat);        

for k = 1:N
    
    ep = epochs_filt{k};         % nChan x nSamples OR 1 x nSamples
    sig = ep;                    % if only one channel; otherwise use ep(pz_idx,:)
    sig = sig(:).';              % ensure row vector
    L = length(sig);

    % 1. Clip windows to signal length
    bs = max(1, baseline_start);
    be = min(L, baseline_end);

    ps = max(1, p300_start);
    pe = min(L, p300_end);

    % 2. Baseline correction
    base_mean = mean(sig(bs:be));
    sig_bc = sig - base_mean;

    % 3. Extract P300 window
    if pe > ps
        p300_seg = sig_bc(ps:pe);
    else
        p300_seg = NaN;    % degenerate trial
    end

    % 4. Feature extraction
    f1 = mean(p300_seg);
    f2 = max(p300_seg);
    f3 = min(p300_seg);
    f4 = trapz(p300_seg);

    X(k,:) = [f1, f2, f3, f4];
end

% (Optional but recommended) Remove any rows with NaNs in features
nan_rows = any(isnan(X), 2);
if any(nan_rows)
    X      = X(~nan_rows, :);
    labels = labels(~nan_rows);
end

% Train / test split
cv = cvpartition(labels, 'HoldOut', 0.3);
Xtrain = X(training(cv), :);
ytrain = labels(training(cv));
Xtest  = X(test(cv), :);
ytest  = labels(test(cv));

% LDA model (replaces SVM)
LDAmodel = fitcdiscr(Xtrain, ytrain, ...
    'DiscrimType','linear', ...   % LDA
    'ClassNames',[0 1], ...
    'Cost', C);                   % your cost matrix

% Predict & evaluate
ypred = predict(LDAmodel, Xtest);
acc   = mean(ypred == ytest);
fprintf('Test accuracy (LDA) = %.3f  --> end_index=%d, C(2,1)=%.2f\n', ...
        acc, end_index, C(2,1));

figure;
confusionchart(ytest, ypred);
title("Confusion Matrix - LDA");
