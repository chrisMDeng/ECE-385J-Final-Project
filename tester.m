path1 = 'C:\Users\perry\OneDrive - The University of Texas at Austin\3 Junior\1st Semester\ECE 385J\BCI Project\sub-P202\ses-S002N2online\eeg\sub-P202_ses-S002N2online_task-Default_run-001_eeg.xdf';
%%
s = load_xdf(path1);

%%
% Load marker, eeg, and timestamp data
eeg = s{1,2}.time_series; % 39 x a array of eeg data (includes EOG and others) (double)
eeg_data = eeg(1:32, :); % 32 x a array of just eeg data (double)
eeg_times = s{1,2}.time_stamps; % 1 x a array of timestamps for eeg data (double)
triggers = s{1,1}.time_series(1,:); % 1 x b array of markers (single)
trigger_times = s{1,1}.time_stamps; % 1 x b array of timestamps for markers (double)

% Load channel labels 
ch_cells = s{1,2}.info.desc.channels.channel;
n = numel(ch_cells);
channel_labels = cell(1, n);

for i = 1:n
    channel_labels{i} = ch_cells{i}.label;
end

%% break triggers and trigger times into epochs

start_indicies = find(triggers == 0);
trigger_groups = cell(1, numel(start_indicies));
trigger_time_groups = cell(1, numel(start_indicies));

for k = 1:numel(start_indicies)
    s = start_indicies(k);

    if k < numel(start_indicies)
        e = start_indicies(k+1) - 1;
    else
        e = numel(triggers);
    end
    
    trigger_groups{k} = triggers(s:e);
    trigger_time_groups{k} = trigger_times(s:e);
end

trigger_groups = trigger_groups(1:end-1); 
trigger_time_groups = trigger_time_groups(1:end-1);

%% break eeg data into epochs

eeg_epochs = cell(1, numel(trigger_groups));

for k = 1:numel(trigger_groups)
    start_time = trigger_time_groups{k}(1);
    disp(trigger_groups{k}(1))
    end_time = trigger_time_groups{k}(2);
    disp(trigger_groups{k}(2))
    mask = (eeg_times >= start_time) & (eeg_times <= end_time);
    eeg_epochs{k} = eeg_data(:, mask);

end

%%

% Initialize match and no-match arrays
match_yes = {};
match_no = {};
nomatch_yes = {};
nomatch_no = {};

% Process each epoch to classify matches and non-matches
for k = 1:numel(eeg_epochs)
    if trigger_groups{k}(end) == 11
        % it's a match
        if trigger_groups{k}(2) == 100
            match_yes{end+1} = eeg_epochs{k}; % Store correct matched epoch
        else
            match_no{end+1} = eeg_epochs{k}; % Store incorrect matched epoch
        end
    else
        % it's not a match
        if trigger_groups{k}(2) == 200
            nomatch_yes{end+1} = eeg_epochs{k}; % Store correct non-matched epoch
        else
            nomatch_no{end+1} = eeg_epochs{k}; % Store incorrect non-matched epoch
        end
    end
end