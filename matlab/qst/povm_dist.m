function p = povm_dist(rho,M)

d = size(M,3);
p = zeros(d,1);
for ii=1:d
    p(ii) = trace(rho*M(:,:,ii));
end

if abs(sum(p) - 1) > 1e-12
    error('Distribution does not sum to one');
end