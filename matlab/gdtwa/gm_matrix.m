function A = gm_matrix(type, alpha, beta, n)

if beta > alpha
    temp = alpha;
    alpha = beta;
    beta = temp;
end

A = zeros(n,n);
switch type
    case 1
        A(beta, alpha) = 1;
        A(alpha, beta) = 1;
        A = (1/sqrt(2))*A;
    case 2
        A(beta, alpha) = 1;
        A(alpha, beta) = -1;
        A = (1/(1i*sqrt(2)))*A;
    case 3
        for i=1:alpha
            A(i,i) = 1;
        end
        A(alpha+1,alpha+1) = -alpha;
        A = (1/sqrt(alpha*(alpha+1)))*A;
    case 4
        A = sqrt(1/n)*eye(n,n);
end