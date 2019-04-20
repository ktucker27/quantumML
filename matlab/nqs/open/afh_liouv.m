function L = afh_liouv(n)

D = 2^n;
eyed = speye(D,D);
H = afh_full(n,1);
L = -1i*(kron(eyed, H) - kron(H, eyed));
