% Make arrays
match_yes = {};
match_no = {};
nomatch_yes = {};
nomatch_no = {};

sub1_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S001N2pre\eeg';
sub1_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S001N2post\eeg';
sub1_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S002N2online\eeg';
sub1_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P201\ses-S003N2online\eeg';

sub2_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S001N2pre\eeg';
sub2_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S001N2post\eeg';
sub2_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S002N2online\eeg';
% issues with this %
sub2_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S003online\eeg';

sub3_pre = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S001N2pre\eeg';
sub3_post = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S001N2post\eeg';
sub3_2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S002online\eeg';
sub3_3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P403\ses-S003N2online\eeg';

folders1 = {sub1_pre, sub1_post, sub1_2, sub1_3};
folders2 = {sub2_pre, sub2_post, sub2_2};
folders3 = {sub3_pre, sub3_post, sub3_2, sub3_3};

folders = {sub3_pre, sub3_post, sub3_2, sub3_3};



for i = 1:numel(folders)
    [a, b, c, d] = load_data(folders1{i});
    match_yes = [match_yes, a];
    match_no = [match_no, b];
    nomatch_yes = [nomatch_yes, c];
    nomatch_no = [nomatch_no, d];
end


%%
a = [match_yes, match_no, nomatch_yes, nomatch_no];

lengths = cellfun(@(x) size(x, 2), a);

figure;
histogram(lengths, 'BinWidth', 10);   % choose bin size you want
xlabel('Length Range (samples)');
ylabel('Number of Cells');
title('Distribution of Cell Lengths');
grid on;


%%

a = match_no;   % or whatever your master array is
lengths = cellfun(@(x) size(x,2), a);

short_arrays = a(lengths < 310 & lengths > 50);        % < 300 samples
long_arrays  = a(lengths >= 310);       % ≥ 300 samples
disp(size(short_arrays))
disp(size(long_arrays))

avg_short = avg_row_26(short_arrays);
avg_long  = avg_row_26(long_arrays);

Fs = 512;   % your sample rate

% Create time vectors
t_short = (0:length(avg_short)-1)/Fs;
t_long  = (0:length(avg_long)-1)/Fs;

figure; hold on;

%plot(t_long,  avg_long,  'LineWidth', 2);
plot(t_short, avg_short, 'LineWidth', 2);

xlabel("Time (s)");
ylabel("Amplitude");
title("Average Row-26 (Long vs Short)");
legend("≥ 300 samples", "< 300 samples");
grid on;

hold off;

%% 

lengths = cellfun(@(x) size(x,2), a);

short_arrays = a(lengths < 310 & lengths > 50);        % < 300 samples
long_arrays  = a(lengths >= 310);       % ≥ 300 samples
disp(size(short_arrays))
disp(size(long_arrays))

avg_short = avg_row_26(short_arrays);
avg_long  = avg_row_26(long_arrays);

Fs = 512;   % your sample rate

% Create time vectors
t_short = (0:length(avg_short)-1)/Fs;
t_long  = (0:length(avg_long)-1)/Fs;

figure; hold on;

plot(t_long,  avg_long,  'LineWidth', 2);
%plot(t_short, avg_short, 'LineWidth', 2);

xlabel("Time (s)");
ylabel("Amplitude");
title("Average Row-26 (Long vs Short)");
legend("≥ 300 samples", "< 300 samples");
grid on;

hold off;


%%
a = long_arrays;

K = numel(a);

minsamp = min(cellfun(@(x) size(x,2), a));
row26 = zeros(K, minsamp);

for k = 1:K
    row26(k, :) = a{k}(26, 1:minsamp);
end

avg_row26 = mean(row26, 1);
%disp(size(avg_row26))

%%

Fs = 512;                 % sample rate (Hz)
x  = avg_row26;
t = (0:length(x)-1)/Fs;     % time axis in seconds

figure;
plot(t, x);
xlabel("Time (s)");
ylabel("Amplitude");
title("Signal vs Time");
grid on;


%% Plot all outcomes (multiple graphs

all_sets = {match_yes, match_no, match_idk, ...
            nomatch_yes, nomatch_no, nomatch_idk};

names = {'match\_yes', 'match\_no', 'match\_idk', ...
         'nomatch\_yes', 'nomatch\_no', 'nomatch\_idk'};

for k = 1:numel(all_sets)
    a = all_sets{k};                     % pick one dataset
    lengths = cellfun(@(x) size(x,2), a);

    figure;
    histogram(lengths, 'BinWidth', 10);
    xlabel('Length Range (samples)');
    ylabel('Number of Cells');
    title(['Distribution of Lengths: ', names{k}]);
    grid on;
end