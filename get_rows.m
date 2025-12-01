function output = get_rows(arrays, end_index)

    % KEEP ONLY CELLS WITH 260 OR MORE SAMPLES
    lens = cellfun(@(x) size(x,2), arrays);  % measure lengths
    arrays = arrays(lens >= end_index);            % keep only >= 260

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
