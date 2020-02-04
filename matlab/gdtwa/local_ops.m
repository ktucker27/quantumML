function [sp, sm, sz] = local_ops(n)

ji = (n-1)/2;
mvec = ji:-1:-ji;
ms = size(mvec,2);
sz = diag(mvec);
sp = circshift(sqrt(diag((ji - mvec).*(ji + mvec + 1))),-1);
sm = circshift(sqrt(diag((ji + mvec).*(ji - mvec + 1))),1);