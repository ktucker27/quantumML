function val = kron_density(mat, a, ~, ~, ~, szl, szr)
% kron_wave Ignores the parameters a and theta and instead pulls the value
%           out of the vector vec in the product basis corresponding to sz

n = size(a,1);

idxl = 0;
delta = 2^(n-1);
for i=1:size(szl,1)
    idxl = idxl - (szl(i) - 1)/2*delta;
    delta = delta/2;
end

idxr = 0;
delta = 2^(n-1);
for i=1:size(szr,1)
    idxr = idxr - (szr(i) - 1)/2*delta;
    delta = delta/2;
end

val = mat(idxl+1, idxr+1);