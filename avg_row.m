function avg_row = avg_row(arrays)
    K = numel(arrays);

    if K == 0
        avg_row = [];
        return;
    end

    minsamp = min(cellfun(@(x) size(x,2), arrays));
    row = zeros(K, minsamp);

    for k = 1:K
        row(k,:) = arrays{k}(1, 1:minsamp);
    end

    avg_row = mean(row, 1);
end