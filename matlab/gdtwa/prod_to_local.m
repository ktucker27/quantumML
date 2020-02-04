function l = prod_to_local(p, n, m)
% prod_to_local converts the index |p> to a vector of its individual base n
% values for each of the m particles. Here we have
%        p = l(m)*n^(m-1) + l(m-1)*n^(m-2) + ... + l(1)*n^0

% Convert from MATLAB one based index to zero based index
p = p - 1;

l = zeros(m,1);
for ii = m:-1:1
    l(ii) = mod(p, n) + 1;
    p = floor(p/n);
end
