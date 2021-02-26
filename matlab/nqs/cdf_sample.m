function sz = cdf_sample(psi, num_samp)
% cdf_sample: Generates num_samp vectors of sampled z-spins from psi,
% returning a [num_samp, n] matrix

tol = 1e-12;

n = log2(size(psi,1));
if floor(n) ~= n
    error('Expected psi to be a [2^n,1] vector');
end

sz = zeros(num_samp, n);

cdf = zeros(size(psi));
for ii=1:size(psi,1)
    if ii == 1
        cdf(ii) = abs(psi(ii))^2;
    else
        cdf(ii) = cdf(ii-1) + abs(psi(ii))^2;
    end
end

if abs(cdf(end) - 1) > tol
    error('Expected psi.^2 to sum to one');
end

for ii=1:num_samp
    eps = rand();
    gt_cdf = find(eps < cdf);
    sz_dec = gt_cdf(1) - 1;
    sz(ii,:) = get_bi(sz_dec, n);
end