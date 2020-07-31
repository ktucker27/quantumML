function [sp, sm, sz, sx, sy] = local_ops(n)

ji = (n-1)/2;
mvec = ji:-1:-ji;

sz = diag(mvec);
sp = circshift(sqrt(diag((ji - mvec).*(ji + mvec + 1))),-1);
sm = circshift(sqrt(diag((ji + mvec).*(ji - mvec + 1))),1);
sx = (1/2)*(sp + sm);
sy = (1/(2*1i))*(sp - sm);