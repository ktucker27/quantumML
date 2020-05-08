function [gamma, err] = slq(A, fun, m, nv)

n = size(A,1);

gamma = 0;

if m >= n
    disp('ERROR - m >= n');
    return;
end

for l=1:nv
    u = get_rad_vec(n);
    v = u/norm(u);
    
    T = lanczos(A, v, m);
    
    [evec, lamda] = eig(T);
    lamda = diag(lamda);
    tau = evec(1,:).';
    
    gamma = gamma + sum((tau.^2).*fun(lamda));
end

gamma = (n/nv)*gamma;

%l = eig(A);
%err = abs(gamma - sum(fun(l)))/sum(fun(l))