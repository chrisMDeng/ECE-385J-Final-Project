fpath1 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\HW1\MI_data_scripts\Subject_006_Session_006_TESS_Online_Visual\Subject_006_TESS_Online__feedback__s006_r001_2021_08_30_163523.gdf';
fpath2 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\HW1\MI_data_scripts\Subject_006_Session_006_TESS_Online_Visual\Subject_006_TESS_Online__feedback__s006_r002_2021_08_30_164143.gdf';
fpath3 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\HW1\MI_data_scripts\Subject_006_Session_006_TESS_Online_Visual\Subject_006_TESS_Online__feedback__s006_r003_2021_08_30_164826.gdf';
fpath4 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\HW1\MI_data_scripts\Subject_006_Session_006_TESS_Online_Visual\Subject_006_TESS_Online__feedback__s006_r004_2021_08_30_165404.gdf';
[s1,h1] = sload(fpath1);
[s2,h2] = sload(fpath2);
[s3,h3] = sload(fpath3);
[s4,h4] = sload(fpath4);
%% checking

% remove last 4 channels
s1 = s1(:, 1:end-4); 
s2 = s2(:, 1:end-4);
s3 = s3(:, 1:end-4);
s4 = s4(:, 1:end-4);

size(s1)
size(s2);
size(s3);
size(s4);
size(h1);
size(h2);
size(h3);
size(h4);

%% temporal filtering

fs = h1.SampleRate;
Nyquist = fs / 2; % Nyquist rate
low = 8; % low cutoff
high = 12; % high cutoff
norm_cutoff_freq = [low high] / Nyquist;

order = 5; % seems to be a good middle ground
type = 'bandpass'; % we are looking at a band from 8-12 Hz
[b,a] = butter(order, norm_cutoff_freq, type);

%filt(b,a,data) or filtfilt(b,a,data)
% use filter because it's causal and could be potentially used for neurofeedback
filtered_data1 = filter(b, a, s1); 
filtered_data2 = filter(b, a, s2); 
filtered_data3 = filter(b, a, s3); 
filtered_data4 = filter(b, a, s4);

%% Just checking if it's actually working
ch = 1;
[pxx_raw, f] = pwelch(s1(:,ch), hamming(2*fs), fs, [], fs);

% PSD
[pxx_filt, f] = pwelch(filtered_data1(:,ch), hamming(2*fs), fs, [], fs);

% Plot
figure;
plot(f,10*log10(pxx_raw),'b','LineWidth',1.5); hold on;
plot(f,10*log10(pxx_filt),'r','LineWidth',1.5);
xlim([0 40]);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend('Raw','Filtered');
title(sprintf('Channel %d PSD before/after filtering', ch));
grid on;

%% spatial filtering
%CAR

%Filter
double_filtered_data1 = filtered_data1 - mean(filtered_data1, 2);
double_filtered_data2 = filtered_data2 - mean(filtered_data2, 2);
double_filtered_data3 = filtered_data3 - mean(filtered_data3, 2);
double_filtered_data4 = filtered_data4 - mean(filtered_data4, 2);

%% Just checking if it's actually working
ch = 1;
[pxx_raw, f] = pwelch(filtered_data1(:,ch), hamming(2*fs), fs, [], fs);

% PSD
[pxx_filt, f] = pwelch(double_filtered_data1(:,ch), hamming(2*fs), fs, [], fs);

% Plot
figure;
plot(f,10*log10(pxx_raw),'b','LineWidth',1.5); hold on;
plot(f,10*log10(pxx_filt),'r','LineWidth',1.5);
xlim([0 40]);
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend('Raw','Filtered');
title(sprintf('Channel %d PSD before/after CAR filtering', ch));
grid on;

%% trial extraction (Get slices)

% Event labels and positions
event_typ1 = h1.EVENT.TYP(:);
event_pos1 = h1.EVENT.POS(:);
event_typ2 = h2.EVENT.TYP(:);
event_pos2 = h2.EVENT.POS(:);
event_typ3 = h3.EVENT.TYP(:);
event_pos3 = h3.EVENT.POS(:);
event_typ4 = h4.EVENT.TYP(:);
event_pos4 = h4.EVENT.POS(:);

start = 1000; % start label

% Mask arrays of where the starts are (for event_typ)
trial_starts1 = (event_typ1==start);
trial_starts2 = (event_typ2==start);
trial_starts3 = (event_typ3==start);
trial_starts4 = (event_typ4==start);

% Array of trial start times (should be 20 each)
starts_pos1 = event_pos1(trial_starts1);
starts_pos2 = event_pos2(trial_starts2);
starts_pos3 = event_pos3(trial_starts3);
starts_pos4 = event_pos4(trial_starts4);

% Convert into single column vector
starts_pos1 = starts_pos1(:);
starts_pos2 = starts_pos2(:);
starts_pos3 = starts_pos3(:);
starts_pos4 = starts_pos4(:);


% Labels
missR = 7692; % miss right label
missL = 7702; % miss left label
hitR = 7693; % hit right label
hitL = 7703; % hit left label

% Mask array of ends (for event_typ)
trial_ends1 = (event_typ1==missL | event_typ1==hitL | event_typ1==missR | event_typ1==hitR);
trial_ends2 = (event_typ2==missL | event_typ2==hitL | event_typ2==missR | event_typ2==hitR);
trial_ends3 = (event_typ3==missL | event_typ3==hitL | event_typ3==missR | event_typ3==hitR);
trial_ends4 = (event_typ4==missL | event_typ4==hitL | event_typ4==missR | event_typ4==hitR);

% Array of end event types (should be 20 each)
trial_type1 = event_typ1(trial_ends1);
trial_type2 = event_typ2(trial_ends2);
trial_type3 = event_typ3(trial_ends3);
trial_type4 = event_typ4(trial_ends4);

% Array of end times (should be 20 each)
trial_pos1 = event_pos1(trial_ends1);
trial_pos2 = event_pos2(trial_ends2);
trial_pos3 = event_pos3(trial_ends3);
trial_pos4 = event_pos4(trial_ends4);

% Mask arrays of left and right (for trial_type (20))
trial_ends_L1 = (trial_type1==missL | trial_type1==hitL);
trial_ends_L2 = (trial_type2==missL | trial_type2==hitL);
trial_ends_L3 = (trial_type3==missL | trial_type3==hitL);
trial_ends_L4 = (trial_type4==missL | trial_type4==hitL);
trial_ends_R1 = ~trial_ends_L1;
trial_ends_R2 = ~trial_ends_L2;
trial_ends_R3 = ~trial_ends_L3;
trial_ends_R4 = ~trial_ends_L4;

% Array of start times (should be 10 each) ///////////////////////////////
starts_pos_L1 = starts_pos1(trial_ends_L1);
starts_pos_R1 = starts_pos1(trial_ends_R1);
starts_pos_L2 = starts_pos2(trial_ends_L2);
starts_pos_R2 = starts_pos2(trial_ends_R2);
starts_pos_L3 = starts_pos3(trial_ends_L3);
starts_pos_R3 = starts_pos3(trial_ends_R3);
starts_pos_L4 = starts_pos4(trial_ends_L4);
starts_pos_R4 = starts_pos4(trial_ends_R4);

% Array of end times (should be 10 each)
ends_pos_L1 = trial_pos1(trial_ends_L1);
ends_pos_R1 = trial_pos1(trial_ends_R1);
ends_pos_L2 = trial_pos2(trial_ends_L2);
ends_pos_R2 = trial_pos2(trial_ends_R2);
ends_pos_L3 = trial_pos3(trial_ends_L3);
ends_pos_R3 = trial_pos3(trial_ends_R3);
ends_pos_L4 = trial_pos4(trial_ends_L4);
ends_pos_R4 = trial_pos4(trial_ends_R4);

% Convert into single column vectors
ends_pos_L1 = ends_pos_L1(:);
ends_pos_R1 = ends_pos_R1(:);
ends_pos_L2 = ends_pos_L2(:);
ends_pos_R2 = ends_pos_R2(:);
ends_pos_L3 = ends_pos_L3(:);
ends_pos_R3 = ends_pos_R3(:);
ends_pos_L4 = ends_pos_L4(:);
ends_pos_R4 = ends_pos_R4(:);
starts_pos_L1 = starts_pos_L1(:);
starts_pos_R1 = starts_pos_R1(:);
starts_pos_L2 = starts_pos_L2(:);
starts_pos_R2 = starts_pos_R2(:);
starts_pos_L3 = starts_pos_L3(:);
starts_pos_R3 = starts_pos_R3(:);
starts_pos_L4 = starts_pos_L4(:)
starts_pos_R4 = starts_pos_R4(:);

% Get number of trials
num_trials_L1 = length(starts_pos_L1);
num_trials_R1 = length(starts_pos_R1);
num_trials_L2 = length(starts_pos_L2);
num_trials_R2 = length(starts_pos_R2);
num_trials_L3 = length(starts_pos_L3);
num_trials_R3 = length(starts_pos_R3);
num_trials_L4 = length(starts_pos_L4)
num_trials_R4 = length(starts_pos_R4);

% Calculate time lengths
time_lengths_L1 = ends_pos_L1 - starts_pos_L1 + 1;
time_lengths_R1 = ends_pos_R1 - starts_pos_R1 + 1;
time_lengths_L2 = ends_pos_L2 - starts_pos_L2 + 1;
time_lengths_R2 = ends_pos_R2 - starts_pos_R2 + 1;
time_lengths_L3 = ends_pos_L3 - starts_pos_L3 + 1;
time_lengths_R3 = ends_pos_R3 - starts_pos_R3 + 1;
time_lengths_L4 = ends_pos_L4 - starts_pos_L4 + 1;
time_lengths_R4 = ends_pos_R4 - starts_pos_R4 + 1;
total_time_lengths_L = cat(1, time_lengths_L1, time_lengths_L2, time_lengths_L3, time_lengths_L4);
total_time_lengths_R = cat(1, time_lengths_R1, time_lengths_R2, time_lengths_R3, time_lengths_R4);
;
clear max % overwriting the max function?

max_lengths = [max(time_lengths_L1), max(time_lengths_L2), max(time_lengths_L3), max(time_lengths_L4), max(time_lengths_R1), max(time_lengths_R2), max(time_lengths_R3), max(time_lengths_R4)]
max_length = max(max_lengths)

total_trials_L = num_trials_L1 + num_trials_L2 + num_trials_L3 + num_trials_L4
total_trials_R = num_trials_R1 + num_trials_R2 + num_trials_R3 + num_trials_R4

%trials_L = cell(total_trials_L,1);
%trials_R = cell(total_trials_R,1);

%% trial extraction (put slices together)

% Gets the number of channels
num_channels = size(double_filtered_data1, 2) 

% % Setup arrays
% data_L = nan(max_length, num_channels, total_trials_L);
% data_R = nan(max_length, num_channels, total_trials_R);
% L_counter = 1;
% R_counter = 1;
% 
% % Assuming they all have the same number of trials:
% % Basically fill in the arrays
% for k = 1:num_trials_L1
%     this_trial = double_filtered_data1(starts_pos_L1(k):ends_pos_L1(k), :);
%     data_L(1:size(this_trial,1), :, L_counter) = this_trial;
%     L_counter = L_counter + 1;
%     this_trial = double_filtered_data2(starts_pos_L2(k):ends_pos_L2(k), :);
%     data_L(1:size(this_trial,1), :, L_counter) = this_trial;
%     L_counter = L_counter + 1;
%     this_trial = double_filtered_data3(starts_pos_L3(k):ends_pos_L3(k), :);
%     data_L(1:size(this_trial,1), :, L_counter) = this_trial;
%     L_counter = L_counter + 1;
%     this_trial = double_filtered_data4(starts_pos_L4(k):ends_pos_L4(k), :);
%     data_L(1:size(this_trial,1), :, L_counter) = this_trial;
%     L_counter = L_counter + 1;
% end
% 
% for k = 1:num_trials_R1
%     this_trial = double_filtered_data1(starts_pos_R1(k):ends_pos_R1(k), :);
%     data_R(1:size(this_trial,1), :, R_counter) = this_trial;
%     R_counter = R_counter + 1;
%     this_trial = double_filtered_data2(starts_pos_R2(k):ends_pos_R2(k), :);
%     data_R(1:size(this_trial,1), :, R_counter) = this_trial;
%     R_counter = R_counter + 1;
%     this_trial = double_filtered_data3(starts_pos_R3(k):ends_pos_R3(k), :);
%     data_R(1:size(this_trial,1), :, R_counter) = this_trial;
%     R_counter = R_counter + 1;
%     this_trial = double_filtered_data4(starts_pos_R4(k):ends_pos_R4(k), :);
%     data_R(1:size(this_trial,1), :, R_counter) = this_trial;
%     R_counter = R_counter + 1;
% end

% Setup arrays
data_L = nan(max_length, num_channels, total_trials_L);
data_R = nan(max_length, num_channels, total_trials_R);
total_time_lengths_L = nan(total_trials_L, 1);
total_time_lengths_R = nan(total_trials_R, 1);
L_counter = 1; R_counter = 1;

K_L = min([num_trials_L1, num_trials_L2, num_trials_L3, num_trials_L4]);  % safe interleave depth

for k = 1:K_L
    % S1
    this_trial = double_filtered_data1(starts_pos_L1(k):ends_pos_L1(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;

    % S2
    this_trial = double_filtered_data2(starts_pos_L2(k):ends_pos_L2(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;

    % S3
    this_trial = double_filtered_data3(starts_pos_L3(k):ends_pos_L3(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;

    % S4
    this_trial = double_filtered_data4(starts_pos_L4(k):ends_pos_L4(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;
end

for k = K_L+1:num_trials_L1
    this_trial = double_filtered_data1(starts_pos_L1(k):ends_pos_L1(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;
end
for k = K_L+1:num_trials_L2
    this_trial = double_filtered_data2(starts_pos_L2(k):ends_pos_L2(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;
end
for k = K_L+1:num_trials_L3
    this_trial = double_filtered_data3(starts_pos_L3(k):ends_pos_L3(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;
end
for k = K_L+1:num_trials_L4
    this_trial = double_filtered_data4(starts_pos_L4(k):ends_pos_L4(k), :);
    data_L(1:size(this_trial,1), :, L_counter) = this_trial;
    total_time_lengths_L(L_counter) = size(this_trial,1);
    L_counter = L_counter + 1;
end

K_R = min([num_trials_R1, num_trials_R2, num_trials_R3, num_trials_R4]);

for k = 1:K_R
    % S1
    this_trial = double_filtered_data1(starts_pos_R1(k):ends_pos_R1(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;

    % S2
    this_trial = double_filtered_data2(starts_pos_R2(k):ends_pos_R2(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;

    % S3
    this_trial = double_filtered_data3(starts_pos_R3(k):ends_pos_R3(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;

    % S4
    this_trial = double_filtered_data4(starts_pos_R4(k):ends_pos_R4(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;
end

for k = K_R+1:num_trials_R1
    this_trial = double_filtered_data1(starts_pos_R1(k):ends_pos_R1(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;
end
for k = K_R+1:num_trials_R2
    this_trial = double_filtered_data2(starts_pos_R2(k):ends_pos_R2(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;
end
for k = K_R+1:num_trials_R3
    this_trial = double_filtered_data3(starts_pos_R3(k):ends_pos_R3(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;
end
for k = K_R+1:num_trials_R4
    this_trial = double_filtered_data4(starts_pos_R4(k):ends_pos_R4(k), :);
    data_R(1:size(this_trial,1), :, R_counter) = this_trial;
    total_time_lengths_R(R_counter) = size(this_trial,1);
    R_counter = R_counter + 1;
end



%% Check correct size
size(data_L)
size(data_R)

%% mu power 

%instantenous power = time samples ^ 2
inst_power_L = data_L .^ 2;
inst_power_R = data_R .^ 2;

%for each trial
    %average power = sum of instanteneous power/n of samples

% Get num trials

num_trials_L = size(inst_power_L, 3)
num_trials_R = size(inst_power_R, 3)

% averages: [average for each channel, trial]
averages_L = nan(num_channels, num_trials_L);
averages_R = nan(num_channels, num_trials_R);

% Take the average and fill in array
for k = 1:num_trials_L
    for j = 1:num_channels
        total = nansum(inst_power_L(:, j, k));
        averages_L(j, k) = total/total_time_lengths_L(k);
    end
end

for k = 1:num_trials_R
    for j = 1:num_channels
        total = nansum(inst_power_L(:, j, k));
        averages_R(j, k) = total/total_time_lengths_L(k);
    end
end
%%
size(averages_L)

%% topoplot
%topoplot(data vector,selectedChannels) % data vector [n of channels x 1]
load('selectedChannels.mat')
topoplot(averages_L(:, 5), selectedChannels)
% Yay it works



%% grand average plot

% Ok, so we have filtered_data1 - filtered_data4 for the unfiltered datasets
% we also have double_filtered_data1 - double_filtered_data4 for the
% filtered datatsets
% The last 0.5 seconds is the last 256 samples since the sample rate is 512
% We need to do it for before and after spatial filtering for each class
% We also have the end times for each class for each run

% So for the times (for both classes)
new_ends_pos_L1 = ends_pos_L1;
new_ends_pos_R1 = ends_pos_R1;
new_ends_pos_L2 = ends_pos_L2;
new_ends_pos_R2 = ends_pos_R2;
new_ends_pos_L3 = ends_pos_L3;
new_ends_pos_R3 = ends_pos_R3;
new_ends_pos_L4 = ends_pos_L4;
new_ends_pos_R4 = ends_pos_R4;
new_starts_pos_L1 = new_ends_pos_L1 - 255;
new_starts_pos_R1 = new_ends_pos_R1 - 255;
new_starts_pos_L2 = new_ends_pos_L2 - 255;
new_starts_pos_R2 = new_ends_pos_R2 - 255;
new_starts_pos_L3 = new_ends_pos_L3 - 255;
new_starts_pos_R3 = new_ends_pos_R3 - 255;
new_starts_pos_L4 = new_ends_pos_L4 - 255;
new_starts_pos_R4 = new_ends_pos_R4 - 255;

% Variables:
% num_channels
% total_trials_L
% total_trials_R
% num_trials_L1 (number of trials in one dataset)
new_length = 256
L_counter = 1;
R_counter = 1;

new_unf_data_L = nan(new_length, num_channels, total_trials_L);
new_unf_data_R = nan(new_length, num_channels, total_trials_R);
new_fil_data_L = nan(new_length, num_channels, total_trials_L);
new_fil_data_R = nan(new_length, num_channels, total_trials_R);

K_L = min([num_trials_L1, num_trials_L2, num_trials_L3, num_trials_L4]);
K_R = min([num_trials_R1, num_trials_R2, num_trials_R3, num_trials_R4]);


N = size(filtered_data1, 1);  % assuming all runs have same length

for k = 1:K_L
    idx = max(1,new_starts_pos_L1(k)) : min(N,new_ends_pos_L1(k));
    this_trial = filtered_data1(idx, :);
    new_unf_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;

    idx = max(1,new_starts_pos_L2(k)) : min(N,new_ends_pos_L2(k));
    this_trial = filtered_data2(idx, :);
    new_unf_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;

    idx = max(1,new_starts_pos_L3(k)) : min(N,new_ends_pos_L3(k));
    this_trial = filtered_data3(idx, :);
    new_unf_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;

    idx = max(1,new_starts_pos_L4(k)) : min(N,new_ends_pos_L4(k));
    this_trial = filtered_data4(idx, :);
    new_unf_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;
end
L_counter = 1;

for k = 1:K_R
    idx = max(1,new_starts_pos_R1(k)) : min(N,new_ends_pos_R1(k));
    this_trial = filtered_data1(idx, :);
    new_unf_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;

    idx = max(1,new_starts_pos_R2(k)) : min(N,new_ends_pos_R2(k));
    this_trial = filtered_data2(idx, :);
    new_unf_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;

    idx = max(1,new_starts_pos_R3(k)) : min(N,new_ends_pos_R3(k));
    this_trial = filtered_data3(idx, :);
    new_unf_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;

    idx = max(1,new_starts_pos_R4(k)) : min(N,new_ends_pos_R4(k));
    this_trial = filtered_data4(idx, :);
    new_unf_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;
end
R_counter = 1;

for k = 1:K_L
    idx = max(1,new_starts_pos_L1(k)) : min(N,new_ends_pos_L1(k));
    this_trial = double_filtered_data1(idx, :);
    new_fil_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;

    idx = max(1,new_starts_pos_L2(k)) : min(N,new_ends_pos_L2(k));
    this_trial = double_filtered_data2(idx, :);
    new_fil_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;

    idx = max(1,new_starts_pos_L3(k)) : min(N,new_ends_pos_L3(k));
    this_trial = double_filtered_data3(idx, :);
    new_fil_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;

    idx = max(1,new_starts_pos_L4(k)) : min(N,new_ends_pos_L4(k));
    this_trial = double_filtered_data4(idx, :);
    new_fil_data_L(1:size(this_trial,1), :, L_counter) = this_trial; L_counter = L_counter + 1;
end

for k = 1:K_R
    idx = max(1,new_starts_pos_R1(k)) : min(N,new_ends_pos_R1(k));
    this_trial = double_filtered_data1(idx, :);
    new_fil_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;

    idx = max(1,new_starts_pos_R2(k)) : min(N,new_ends_pos_R2(k));
    this_trial = double_filtered_data2(idx, :);
    new_fil_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;

    idx = max(1,new_starts_pos_R3(k)) : min(N,new_ends_pos_R3(k));
    this_trial = double_filtered_data3(idx, :);
    new_fil_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;

    idx = max(1,new_starts_pos_R4(k)) : min(N,new_ends_pos_R4(k));
    this_trial = double_filtered_data4(idx, :);
    new_fil_data_R(1:size(this_trial,1), :, R_counter) = this_trial; R_counter = R_counter + 1;
end

%% Grand Averages:

g_averages_unf_L = nan(num_channels,1);
g_averages_unf_R = nan(num_channels,1);
g_averages_fil_L = nan(num_channels,1);
g_averages_fil_R = nan(num_channels,1);

for k = 1:num_channels
    g_averages_unf_L(k) = mean(new_unf_data_L(:, k, :), 'all', 'omitnan');
    g_averages_unf_R(k) = mean(new_unf_data_R(:, k, :), 'all', 'omitnan');
    g_averages_fil_L(k) = mean(new_fil_data_L(:, k, :), 'all', 'omitnan');
    g_averages_fil_R(k) = mean(new_fil_data_R(:, k, :), 'all', 'omitnan');
end

%%
size(g_averages_unf_L)
size(new_fil_data_R)
limits = [-0.07, 0.07];

%% Grand Average Topoplot Unfiltered Left
topoplot(g_averages_unf_L, selectedChannels, 'maplimits',limits); % or [lo hi]
cb = colorbar; 
cb.Label.String = '\muV';

%% Grand Average Topoplot Unfiltered Right
topoplot(g_averages_unf_R, selectedChannels, 'maplimits',limits); % or [lo hi]
cb = colorbar;
cb.Label.String = '\muV';

%% Grand Average Topoplot Filtered Left
topoplot(g_averages_fil_L, selectedChannels, 'maplimits',limits); % or [lo hi]
cb = colorbar;
cb.Label.String = '\muV';

%% Grand Average Topoplot Filtered Right
topoplot(g_averages_fil_R, selectedChannels, 'maplimits',limits); % or [lo hi]
cb = colorbar;
cb.Label.String = '\muV'; 
