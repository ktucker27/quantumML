function [scalar, idx] = gm_comm(mu, nu, n)

sgn = 1;
if mu > nu
    sgn = -1;
    temp = mu;
    mu = nu;
    nu = temp;
end

[type1, alpha1, beta1] = gm_idx(mu, n);
[type2, alpha2, beta2] = gm_idx(nu, n);

scalar = [];
idx = [];

if mu == nu
    return
end

switch type1
    case 1
        switch type2
            case 1
                scalar = 1i/sqrt(2);
                if beta1 == beta2
                    idx = gm_local_idx(2, alpha1, alpha2, n);
                elseif alpha1 == alpha2
                    idx = gm_local_idx(2, beta1, beta2, n);
                elseif beta1 == alpha2
                    idx = gm_local_idx(2, alpha1, beta2, n);
                elseif alpha1 == beta2
                    idx = gm_local_idx(2, beta1, alpha2, n);
                end
            case 2
                if beta1 == beta2
                    if alpha1 ~= alpha2
                        scalar = -1i/sqrt(2);
                        idx = gm_local_idx(1, alpha1, alpha2, n);
                    else
                        scalar = zeros(n - max([beta1-1,1]),1);
                        idx = zeros(n - max([beta1-1,1]),1);
                        ii = 1;
                        for gamma = max([beta1-1,1]):n-1
                            tr = 0;
                            if gamma == beta1-1
                                tr = tr - gamma;
                            end
                            
                            if gamma == alpha1-1
                                tr = tr + gamma;
                            end
                            
                            if gamma >= beta1
                                tr = tr + 1;
                            end
                            
                            if gamma >= alpha1
                                tr = tr - 1;
                            end
                            
                            tr = (1i/sqrt(gamma*(gamma+1)))*tr;
                            scalar(ii,1) = tr;
                            idx(ii,1) = gm_local_idx(3, gamma, 0, n);
                            ii = ii + 1;
                        end
                        %[scalar, idx] = gm_expand([beta1, beta2;alpha1, alpha2], [1; -1], n);
                        %scalar = 1i*scalar;
                    end
                elseif alpha1 == alpha2
                    scalar = 1i/sqrt(2);
                    idx = gm_local_idx(1, beta1, beta2, n);
                elseif beta1 == alpha2
                    scalar = 1i/sqrt(2);
                    idx = gm_local_idx(1, beta2, alpha1, n);
                elseif alpha1 == beta2
                    scalar = -1i/sqrt(2);
                    idx = gm_local_idx(1, beta1, alpha2, n);
                end
            case 3
                if alpha2 == alpha1 - 1
                    scalar = -1i*alpha1/sqrt(alpha1*(alpha1-1));
                    idx = gm_local_idx(2, alpha1, beta1, n);
                elseif alpha2 < alpha1 - 1 && alpha2 >= beta1
                    scalar = -1i/sqrt(alpha2*(alpha2+1));
                    idx = gm_local_idx(2, alpha1, beta1, n);
                elseif alpha2 == beta1 - 1
                    scalar = 1i*(beta1 - 1)/sqrt(beta1*(beta1-1));
                    idx = gm_local_idx(2, alpha1, beta1, n);
                end
        end
    case 2
        switch type2
            case 2
                scalar = 1i/sqrt(2);
                if beta1 == beta2
                    idx = gm_local_idx(2, alpha1, alpha2, n);
                elseif alpha1 == alpha2
                    idx = gm_local_idx(2, beta1, beta2, n);
                elseif beta1 == alpha2
                    scalar = -1*scalar;
                    idx = gm_local_idx(2, alpha1, beta2, n);
                elseif alpha1 == beta2
                    scalar = -1*scalar;
                    idx = gm_local_idx(2, beta1, alpha2, n);
                end
            case 3
                if alpha2 == alpha1 - 1
                    scalar = 1i*alpha1/sqrt(alpha1*(alpha1-1));
                    idx = gm_local_idx(1, alpha1, beta1, n);
                elseif alpha2 < alpha1 - 1 && alpha2 >= beta1
                    scalar = 1i/sqrt(alpha2*(alpha2+1));
                    idx = gm_local_idx(1, alpha1, beta1, n);
                elseif alpha2 == beta1 - 1
                    scalar = -1i*(beta1 - 1)/sqrt(beta1*(beta1-1));
                    idx = gm_local_idx(1, alpha1, beta1, n);
                end
        end
end

scalar = sgn*scalar;