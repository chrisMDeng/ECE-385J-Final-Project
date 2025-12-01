function [match_yes, match_no, nomatch_yes, nomatch_no] = load_data(folder_path)
    
    % Get all file paths in a folder
    paths = {};
    files = dir(fullfile(folder_path, '*'));   % list everything
    
    for k = 1:length(files)
        fname = files(k).name;
    
        % skip '.' and '..'
        if files(k).isdir
            continue
        end
    
        fullpath = fullfile(folder_path, fname);
    
        paths{end+1} = fullpath;
    end


    % Initialize match and nomatch arrays
    match_yes = {};
    match_no = {};
    nomatch_yes = {};
    nomatch_no = {};


    for p = 1:numel(paths)

        % Load xdf data
        s = load_xdf(paths{p});
    
        
        % Load marker, eeg, and timestamp data
        eeg = s{1,2}.time_series; % 39 x a array of eeg data (includes EOG and others) (double)
        if size(eeg) ~= 39
            disp(paths{p})
        end
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
        
    
        % break triggers and trigger times into epochs
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
        
        % Discard first one because it's not 2-back and last one because of
        % missing data
        trigger_groups = trigger_groups(2:end-1); 
        trigger_time_groups = trigger_time_groups(2:end-1);
        
    
        % break eeg data into epochs
        eeg_epochs = cell(1, numel(trigger_groups));
        
        for k = 1:numel(trigger_groups)
            start_time = trigger_time_groups{k}(1);
            %disp(trigger_groups{k}(1))
            end_time = trigger_time_groups{k}(2);
            %disp(trigger_groups{k}(2))
            mask = (eeg_times >= start_time) & (eeg_times <= end_time);
            eeg_epochs{k} = eeg_data(:, mask);
        
        end
    
    
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
    end
end


