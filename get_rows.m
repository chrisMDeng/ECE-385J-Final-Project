function output = get_rows(arrays, end_index)
    %filters out samples based on how many samples they have
    %selects one channel

    % KEEP ONLY CELLS WITH end_index OR MORE SAMPLES
    lens = cellfun(@(x) size(x,2), arrays);  % measure lengths
    arrays = arrays(lens >= end_index);            % keep only >= end_index

    n = numel(arrays);

    if n == 0
        output = [];
        return;
    end

    output = cell(1,n);
    for k = 1:n
        output{k} = arrays{k}(26, :);  % no 1:end needed
    end
end
