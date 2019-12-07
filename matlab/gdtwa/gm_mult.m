function [c, delta] = gm_mult(type, alpha, beta, gamma, n)

if beta > alpha
    temp = alpha;
    alpha = beta;
    beta = temp;
end

if type <= 2 && gamma ~= alpha && gamma ~= beta
    c = 0;
    delta = -1;
    return;
end

switch type
    case 1
        c = 1/sqrt(2);
        if gamma == alpha
            delta = beta;
        else
            delta = alpha;
        end
    case 2
        c = -1i/sqrt(2);
        if gamma == alpha
            delta = beta;
        else
            delta = alpha;
            c = -1*c;
        end
    case 3
        if gamma <= alpha
            c = 1/sqrt(alpha*(alpha+1));
            delta = gamma;
        elseif gamma == alpha + 1
            c = -alpha/sqrt(alpha*(alpha+1));
            delta = gamma;
        else
            c = 0;
            delta = -1;
        end
    case 4
        c = 1/sqrt(n);
        delta = gamma;
end