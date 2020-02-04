function [pass, norms] = gm_mult_test(n)
%GM_MULT_TEST An exhaustive test of gm_mult, multiplying each GM matrix by
%each local basis vector and comparing it to the full matrix expansion

pass = 1;

TOL = 1e-12;

norms = zeros(4,n,n,n);
for type=1:4
    for alpha=1:n
        if type == 3 && alpha == n
            continue
        end
        
        if type == 4 && alpha > 1
            continue
        end
        for beta=1:alpha
            if type >= 3 && beta > 1
                continue
            end
            
            if type < 3 && beta >= alpha
                continue
            end
            
            A = gm_matrix(type, alpha, beta, n);
            for gamma=1:n
                [c, delta] = gm_mult(type, alpha, beta, gamma, n);
                v = zeros(n,1);
                v(gamma) = 1;
                v = A*v;
                v2 = zeros(n,1);
                if abs(c) > TOL
                    v2(delta) = c;
                end
                nm = norm(v - v2);
                norms(type, alpha, beta, gamma) = nm;
                if nm > TOL
                    pass = 0;
                    disp(['FAIL: (', num2str(type), ', ', num2str(alpha), ', ', num2str(beta), ', ', num2str(gamma), ')  ', num2str(nm)]);
                %else
                %    disp(['PASS: (', num2str(type), ', ', num2str(alpha), ', ', num2str(beta), ', ', num2str(gamma), ')  ', num2str(nm)]);
                end
            end
        end
    end
end

if pass ~= 0
    disp('gm_mult_test PASSED');
end