function mu = gm_local_idx(type, alpha, beta, n)

if beta > alpha
    temp = alpha;
    alpha = beta;
    beta = temp;
end

mu = 0;
if type > 1
    mu = mu + n*(n-1)/2;
else
    mu = mu + (beta-1)*n + alpha - beta*(beta+1)/2;
    return
end

if type > 2
    mu = mu + n*(n-1)/2;
else
    mu = mu + (beta-1)*n + alpha - beta*(beta+1)/2;
    return
end

if type > 3
    mu = mu + n;
else
    mu = mu + alpha;
    return
end