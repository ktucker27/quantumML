function b = lanczos_expm(A, v, m)

n = size(A,1);

if m >= n
    error('Need m < n');
end
    
[T, Q] = lanczos(A, v, m);
    
[evec, lamda] = eig(T);

b = Q*evec*diag(exp(diag(lamda)))*evec(1,:)';
