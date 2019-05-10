function [a, b, w] = unpack_tvmc(y, n, m)

a = y(1:n);
b = y(n+1:n+m);
w = reshape(y(n+m+1:end),[m,n]);