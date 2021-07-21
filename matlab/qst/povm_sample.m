function samples = povm_sample(rho, M, local_idx, num_samp)
% povm_sample: Generates num_samp vectors of samples from POVM M and state
% rho, returning a [num_samp, n] categorical of values from 1-d equal to
% local_idx(idx,:) where the idx outcome corresponding to operator
% M(:,:,idx) is sampled

tol = 1e-12;

samples = zeros(num_samp, n);

% Generate the probability distribution for this state and POVM
p = povm_dist(rho, M);

% Get the CDF for the distribution
cdf = zeros(size(p));
for ii=1:size(p,1)
    if ii == 1
        cdf(ii) = p(ii);
    else
        cdf(ii) = cdf(ii-1) + p(ii);
    end
end

if abs(cdf(end) - 1) > tol
    error('Expected p to sum to one');
end

% Sample from the CDF num_samp times recording the corresponding local POVM
% indices in the output variable samples
for ii=1:num_samp
    eps = rand();
    gt_cdf = find(eps < cdf);
    meas_idx = gt_cdf(1);
    samples(ii,:) = local_idx(meas_idx,:);
end