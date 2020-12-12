function val = kron_wave(vec, a, ~, sz)
% kron_wave Ignores the parameters a and theta and instead pulls the value
%           out of the vector vec in the product basis corresponding to sz

n = size(a,1);

idx = 0;
delta = 2^(n-1);
for i=1:size(sz,1)
    idx = idx - (sz(i) - 1)/2*delta;
    delta = delta/2;
end

val = vec(idx+1);