function [T,V] = povm_overlap(M)

d = size(M,3);
T = zeros(d,d);
for ii=1:d
    for jj=1:d
        T(ii,jj) = trace(M(:,:,ii)*M(:,:,jj));
    end
end

[~, ~, sz, sx, sy] = local_ops(2);
sx = 2*sx;
sy = 2*sy;
sz = 2*sz;
V = zeros(d,d);
for ii=1:d
    V(ii,1) = trace(M(:,:,ii)*eye(2));
    V(ii,2) = trace(M(:,:,ii)*sx);
    V(ii,3) = trace(M(:,:,ii)*sy);
    V(ii,4) = trace(M(:,:,ii)*sz);
end