% Load data
data = load('ErrP_data_scripts\ErrP_data_HW1.mat');
channels = load('ErrP_data_scripts\ErrP_channels.mat');

%% Exploring data
% 1) See fields
fieldnames(data)

% Look at trainingEpochs
T = data.trainingEpochs;
fieldnames(T) 

% Look at size and variable type for each of the variables in it
T = data.trainingEpochs;
fn = fieldnames(T);
for i = 1:numel(fn)
    x = T.(fn{i});
    fprintf('%-24s  %s   %s\n', fn{i}, mat2str(size(x)), class(x));
end

% Data output:
% 
% rotation_data             [1024 32 320]   double
% label                     [320 1]   double
% magnitude                 [320 1]   double
% fileID                    [320 1]   double
% sessionID                 [320 1]   double

% Since there's 2 seconds [-1, 1] and 1024 data points, it seems like the
% sampling rate is 512Hz. 
% 320 trials
% Already filtered in the theta band YAY!!!

%% Computer the Grand Average for each magnitude

% Get EEG data
s = data.trainingEpochs.rotation_data; 
size(s); % Outputs 1024 x 32 x 320

mask0 = (data.trainingEpochs.magnitude == 0);
mask3 = (data.trainingEpochs.magnitude == 3);
mask6 = (data.trainingEpochs.magnitude == 6);
mask9 = (data.trainingEpochs.magnitude == 9);
mask12 = (data.trainingEpochs.magnitude == 12);

sum(mask0); % 160 trials
sum(mask3); % 40 trials
sum(mask6); % 40 trials
sum(mask9); % 40 trials
sum(mask12); % 40 trials

data0 = s(:, :, mask0);
data3 = s(:, :, mask3);
data6 = s(:, :, mask6);
data9 = s(:, :, mask9);
data12 = s(:, :, mask12);

% Take Average (now size is 1024 x 32)
trials_average0 = mean(data0, 3);
trials_average3 = mean(data3, 3);
trials_average6 = mean(data6, 3);
trials_average9 = mean(data9, 3);
trials_average12 = mean(data12, 3);

% Figure out which channel is Cz 
help = channels.params.chanlocs;
disp(help(15));
% Seems like Cz is channel 15

plotter0 = trials_average0(1:1024, 15);
plotter3 = trials_average3(1:1024, 15);
plotter6 = trials_average6(1:1024, 15);
plotter9 = trials_average9(1:1024, 15);
plotter12 = trials_average12(1:1024, 15);


Y= [plotter0, plotter3, plotter6, plotter9, plotter12];
l = [0, 3, 6, 9, 12];

t = (0:1023)/512;


for i = 1:size(Y,2)      % i = 1..5
    figure(i);                          % new window each loop
    plot(t, Y(:,i), 'LineWidth', 1.6);
    grid on; xlabel('Time (s)'); ylabel('Amplitude');
    title(sprintf('Magnitude: %d', l(i)));
    xlim([t(1) t(end)]);
end

%% Find Peaks

% Negative Peak Ranges:
% 0: 0.2-0.3
% 3: 0.2-0.3
% 6: 0.3-0.4
% 9: 0.2-0.3
% 12: 0.2-0.3

% Positive Peak Ranges:
% 0: 0.5-0.6
% 3: 0.4-0.5
% 6: 0.4-0.5
% 9: 0.4-0.5
% 12: 0.3-0.4

negs = [1.2, 1.2, 1.3, 1.2, 1.2];
pos = [1.5, 1.4, 1.4, 1.4, 1.3];
n_peak = nan(5, 2);
p_peak = nan(5, 2);
fs = 512;

for i=1:5
    neg0 = round(negs(i)*fs);
    neg1 = round((negs(i)+0.1)*fs);
    pos0 = round(pos(i)*512);
    pos1 = round((pos(i)+0.1)*512);
    [n_peak(i, 1), n_peak(i, 2)] = min(Y(neg0:neg1, i));
    n_peak(i, 2) = neg0 + (n_peak(i, 2) - 1)
    [p_peak(i, 1), p_peak(i, 2)] = max(Y(pos0:pos1, i));
    p_peak(i, 2) = pos0 + (p_peak(i, 2) - 1)
end

%% Load Channel Locations and prep vectors

size(trials_average0)
Z = cat(3, trials_average0, trials_average3, trials_average6, trials_average9, trials_average12);

pos_vectors = nan(32, 5);
neg_vectors = nan(32, 5);

for i=1:5
    a = Z(p_peak(i, 2), :, i);
    a = a(:);
    size(a);
    pos_vectors(:, i) = a;

    b = Z(n_peak(i, 2), :, i);
    b = b(:);
    size(b);
    neg_vectors(:, i) = b;
end


%% topoplot
%topoplot(data vector,selectedChannels) % data vector [n of channels x 1]
%load('selectedChannels.mat')
%topoplot(pos_vectors(:, 5), selectedChannels)
limits = [-3, 8]
topoplot(neg_vectors(:, 4), channels.params.chanlocs, 'maplimits', 'absmax'); % or [lo hi]
%topoplot(pos_vectors(:, 5), channels.params.chanlocs, 'maplimits', limits); % or [lo hi]
cb = colorbar;
cb.Label.String = '\muV';
% Yay it works

%% With CAR:

% Original data:
% [1024 x 32 x trials]
%data0 = s(:, :, mask0);
%data3 = s(:, :, mask3);
%data6 = s(:, :, mask6);
%data9 = s(:, :, mask9);
%data12 = s(:, :, mask12);

%CAR (find mean of all channels for each time point)

%Filter
car_data0 = data0 - mean(data0, 2);
car_data3 = data3 - mean(data3, 2);
car_data6 = data6 - mean(data6, 2);
car_data9 = data9 - mean(data9, 2);
car_data12 = data12 - mean(data12, 2);

%% Grand Average (average over trials for each magnitude)
ga_data0 = mean(car_data0, 3);
ga_data3 = mean(car_data3, 3);
ga_data6 = mean(car_data6, 3);
ga_data9 = mean(car_data9, 3);
ga_data12 = mean(car_data12, 3);
size(ga_data0)

%% Prep Vectors
D = cat(3, ga_data0, ga_data3, ga_data6, ga_data9, ga_data12);

pos_car_vectors = nan(32, 5);
neg_car_vectors = nan(32, 5);

for i=1:5
    a = D(p_peak(i, 2), :, i);
    a = a(:);
    size(a);
    pos_car_vectors(:, i) = a

    b = D(n_peak(i, 2), :, i);
    b = b(:);
    size(b);
    neg_car_vectors(:, i) = b
end

%% topoplot
%topoplot(data vector,selectedChannels) % data vector [n of channels x 1]
%topoplot(pos_vectors(:, 5), selectedChannels)
limits = [-3, 8]
filtered_limits = [-2, 4]; % neg
%topoplot(pos_car_vectors(:, 1), channels.params.chanlocs, 'maplimits', 'absmax'); % or [lo hi]
topoplot(pos_car_vectors(:, 5), channels.params.chanlocs, 'maplimits', filtered_limits); % or [lo hi]
cb = colorbar;
cb.Label.String = '\muV';

%% Computer the Grand Average for each magnitude

% Get EEG data
s = data.trainingEpochs.rotation_data; 
size(s); % Outputs 1024 x 32 x 320

mask_cor = (data.trainingEpochs.label == 0);
mask_err = (data.trainingEpochs.label == 1);

sum(mask_cor); % 160 trials
sum(mask_err); % 160 trials

% size is 1024 x 32 x 160
data_cor = s(:, :, mask_cor);
data_err = s(:, :, mask_err);
size(data_cor);
size(data_err);

% Take Average (now size is 1024 x 32)
trials_average_err = mean(data0, 3);
trials_average_cor = mean(data3, 3);

% 163840 x 32
reshaped_data_cor = reshape(permute(data_cor,[1 3 2]), 1024*160, 32);
reshaped_data_err = reshape(permute(data_err,[1 3 2]), 1024*160, 32);

% 163840 x 32
repeated_average_err = repmat(trials_average_err, 160, 1);
repeated_average_cor = repmat(trials_average_cor, 160, 1);

%% CCA
%maximize correlation between the single-trial data and the class grand average template

[spatialFilter_cor,~] =  canoncorr(reshaped_data_cor, repeated_average_cor); % spatial filter [n channels x n components]
[spatialFilter_err,~] =  canoncorr(reshaped_data_err, repeated_average_err); % spatial filter [n channels x n components]

% size 32 x 32 (each column is the 32 x 1 weights)
size(spatialFilter_cor);
size(spatialFilter_err);

weights_cor = sum(spatialFilter_cor(:, 1:5), 2);
weights_err = sum(spatialFilter_err(:, 1:5), 2);

size(weights_cor)


%% topoplot
filtered_limits = [-0.6, 0.6]; % neg
topoplot(weights_cor(:), channels.params.chanlocs, 'maplimits', 'absmax'); % or [lo hi]
%topoplot(weights_err(:), channels.params.chanlocs, 'maplimits', filtered_limits); % or [lo hi]
cb = colorbar;
cb.Label.String = '\muV';