function [psp, psm, psz] = prod_ops(part, n, m)
%PROD_OPS Returns sparse matrix representations of the full product space
%operators corresponding to the +/-/z operator for a given particle

[sp, sm, sz] = local_ops(n);

psp = local_op_to_prod(sp, part, n, m);
psm = local_op_to_prod(sm, part, n, m);
psz = local_op_to_prod(sz, part, n, m);
