function idx = get_trunc_idx(v, eps)
v2 = v.^2;
d = sum(v2);
sv = zeros(size(v));
for ii=1:size(v,1)
    sv(ii,1) = sqrt(sum(v2(end:-1:ii))/d);
end
idxvec = find(sv < eps);
if size(idxvec,1) == 0
    idx = size(v,1);
else
    idx = idxvec(1)-1;
end

if idx == 0
    error('Found zero truncation index');
end
end