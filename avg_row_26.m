function avg_row26 = avg_row26(arrays)
    K = numel(arrays);

    if K == 0
        avg_row26 = [];
        return;
    end

    minsamp = min(cellfun(@(x) size(x,2), arrays));
    row26 = zeros(K, minsamp);

    for k = 1:K
        row26(k,:) = arrays{k}(26, 1:minsamp);
    end

    avg_row26 = mean(row26, 1);
end