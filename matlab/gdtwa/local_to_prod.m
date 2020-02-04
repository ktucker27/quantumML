function p = local_to_prod(l, n)
m = size(l,1);
p = 0;
for ii = 1:m
    p = p + (l(ii)-1)*n^(m - ii);
end

% Convert from zero based index to MATLAB one based index
p = p + 1;