function [u, w] = gm_expand(O, n, m)
% gm_expand Calculate the first and second order generalized Gell-Mann
% coefficients for the operator O. Assumes O does not have any higher order
% terms. O is a sparse matrix in the full tensor product space basis
%
% INPUT: O - operator to decompose
%        n - number of levels per particle (dimension of local Hilbert
%            space)
%        m - number of particles
%
% OUTPUT: u - first order GM coefficients
%         w - second order GM coefficients

TOL = 1.0e-12;

d = n^2;

u = sparse(d*m,1);
w = sparse(d*m, d*m);

[is, js, vs] = find(O);

for ii = 1:size(is,1)
    % TODO - Speed this up
    li = prod_to_local(is(ii), n, m);
    
    for part_idx1 = 1:m
        for mu = 1:d
            [type, alpha, beta] = gm_idx(mu, n);
            [c, delta] = gm_mult(type, alpha, beta, li(part_idx1), n);
            
            if abs(c) < TOL
                continue
            end
            
            li2 = li;
            li2(part_idx1) = delta;
            p = local_to_prod(li2, n);
            if p ~= js(ii)
                continue
            end
            
            u((part_idx1-1)*d + mu,1) = u((part_idx1-1)*d + mu,1) + vs(ii)*c;
        end
        
        for part_idx2 = 1:m
            if part_idx2 == part_idx1
                continue
            end
            
            for mu = 1:d-1
                [type1, alpha1, beta1] = gm_idx(mu, n);
                [c1, delta1] = gm_mult(type1, alpha1, beta1, li(part_idx1), n);
                
                if abs(c1) < TOL
                    continue
                end
                
                for nu = 1:d-1
                    [type2, alpha2, beta2] = gm_idx(nu, n);
                    [c2, delta2] = gm_mult(type2, alpha2, beta2, li(part_idx2), n);
                    
                    if abs(c2) < TOL
                        continue
                    end
                    
                    li2 = li;
                    li2(part_idx1) = delta1;
                    li2(part_idx2) = delta2;
                    p = local_to_prod(li2, n);
                    if p ~= js(ii)
                        continue
                    end
                    
                    w((part_idx1-1)*d + mu, (part_idx2-1)*d + nu) = w((part_idx1-1)*d + mu, (part_idx2-1)*d + nu) + vs(ii)*c1*c2;
                end
            end
        end
    end
end

% TODO - Figure out the w normalization and how to avoid over/underflow for
% larger systems
u = u/n^(m-1);
%w = w/n^(2*(m-1))*n^m/2;
w = w/(n^(m-2)*2);