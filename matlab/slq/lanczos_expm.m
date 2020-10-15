function b = lanczos_expm(A, v, m, fun)

if nargin < 4
    fun = @(x)(exp(x));
end

n = size(A,1);

if m >= n
    error('Need m < n');
end
    
[T, Q] = lanczos(A, v, m);
    
[evec, lamda] = eig(T);

b = Q*evec*diag(fun(diag(lamda)))*evec(1,:)';
