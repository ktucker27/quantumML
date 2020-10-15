function [T, Q, alphas, betas] = lanczos(A, b, numsteps)

T = zeros(numsteps+1, numsteps+1);
Q = zeros(size(A,1),numsteps+1);
Q(:,1) = b/norm(b);
alphas = zeros(numsteps+1,1);
betas = zeros(numsteps,1);
beta = 0;
for i=1:numsteps
    z = A*Q(:,i);
    alpha = Q(:,i)'*z;
    z = z - alpha*Q(:,i);
    if i > 1
        z = z - beta*Q(:,i-1);
    end
    beta = norm(z);
    if abs(beta) < 1e-10
        %disp(['WARNING - Found linear dependence on step ', num2str(i)]);
        numsteps = i-1;
        Q = Q(:,1:numsteps+1);
        T = T(1:numsteps+1, 1:numsteps+1);
        break;
    end
    Q(:,i+1) = z/beta;
    
    T(i,i) = alpha;
    T(i,i+1) = beta;
    T(i+1,i) = beta;
    
    alphas(i,1) = alpha;
    betas(i,1) = beta;
end

alpha = Q(:,numsteps+1)'*A*Q(:,numsteps+1);
T(numsteps+1, numsteps+1) = alpha;
alphas(numsteps+1,1) = alpha;