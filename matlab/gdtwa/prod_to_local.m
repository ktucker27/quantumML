function l = prod_to_local(p, n, m)
l = zeros(m,1);
for ii = 1:m
    l(ii) = mod(p, n);
    p = floor(p/n);
end
