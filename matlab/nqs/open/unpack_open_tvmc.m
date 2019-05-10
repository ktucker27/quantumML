function [a, b, c, w, u] = unpack_open_tvmc(y, n, m, l)

sidx = 1;
a = y(sidx:sidx+n-1,1) + 1i*y(sidx+n:sidx+2*n-1,1);
sidx = sidx + 2*n;
b = y(sidx:sidx+m-1,1) + 1i*y(sidx+m:sidx+2*m-1,1);
sidx = sidx + 2*m;
c = y(sidx:sidx+l-1,1);
sidx = sidx + l;
w = reshape(y(sidx:sidx+m*n-1,1), [m,n]) + 1i*reshape(y(sidx+m*n:sidx+2*m*n-1,1), [m,n]);
sidx = sidx + 2*m*n;
u = reshape(y(sidx:sidx+l*n-1,1), [l,n]) + 1i*reshape(y(sidx+l*n:end,1), [l,n]);