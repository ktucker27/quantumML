function [type, alpha, beta] = gm_idx(mu, n)

if mu <= n*(n-1)/2
    type = 1;
elseif mu <= n*(n-1)
    type = 2;
    mu = mu - n*(n-1)/2;
elseif mu < n*n
    type = 3;
    alpha = mu - n*(n-1);
    beta = 0;
else
    type = 4;
    alpha = 0;
    beta = 0;
end

if type == 1 || type == 2
    alpha = mu;
    beta = 1;
    row_size = n - 1;
    while alpha > row_size
        alpha = alpha - row_size;
        row_size = row_size - 1;
        beta = beta + 1;
    end
    
    alpha = alpha + beta;
end