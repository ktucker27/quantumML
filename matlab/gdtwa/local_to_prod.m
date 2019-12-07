function p = local_to_prod(l, n)
m = size(l,1);
p = 0;
for ii = 1:m
    p = p + l(ii)*n^(ii-1);
end
