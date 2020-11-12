function b = lanczos_expm_mps(L, R, ops, v, m, fun)

if nargin < 6
    fun = @(x)(exp(x));
end

n = prod(v.dims());

if m >= n
    error('Need m < n');
end
    
[T, Q] = lanczos_mps(L, R, ops, v, m);
    
[evec, lamda] = eig(T);

b = Q*evec*diag(fun(diag(lamda)))*evec(1,:)';
