function pass = gm_expand_test(n, m)

pass = 1;

if test_single(n, m) == 0
    pass = 0;
end

if test_double(n, m) == 0
    pass = 0;
end

if pass ~= 0
    disp(['gm_expand_test(n = ', num2str(n), ', m = ', num2str(m), ') PASSED']);
else
    disp(['gm_expand_test(n = ', num2str(n), ', m = ', num2str(m), ') FAILED']);
end
end

function pass = test_single(n, m)
pass = 1;
for part=1:m
    [psp, psm, psz] = prod_ops(part, n, m);
    nm = op_compare(psp, n, m);
    if nm > 1e-12
        pass = 0;
        disp(['FAIL: op_compare(psp, ', num2str(n), ', ', num2str(m), '), part=', num2str(part), ', ', num2str(nm)]);
    end
    
    nm = op_compare(psm, n, m);
    if nm > 1e-12
        pass = 0;
        disp(['FAIL: op_compare(psm, ', num2str(n), ', ', num2str(m), '), part=', num2str(part), ', ', num2str(nm)]);
    end
    
    nm = op_compare(psz, n, m);
    if nm > 1e-12
        pass = 0;
        disp(['FAIL: op_compare(psz, ', num2str(n), ', ', num2str(m), '), part=', num2str(part), ', ', num2str(nm)]);
    end
end
end

function pass = test_double(n, m)
pass = 1;
for part1=2:m
    [psp1, psm1, psz1] = prod_ops(part1, n, m);
    for part2=1:part1-1
        [psp2, psm2, psz2] = prod_ops(part2, n, m);
        
        nm = op_compare(psp1*psp2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psp1*psp2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psp1*psm2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psp1*psm2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psp1*psz2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psp1*psz2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psm1*psp2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psm1*psp2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psm1*psm2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psm1*psm2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psm1*psz2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psm1*psz2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psz1*psp2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psz1*psp2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psz1*psm2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psz1*psm2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
        
        nm = op_compare(psz1*psz2, n, m);
        if nm > 1e-12
            pass = 0;
            disp(['FAIL: op_compare(psz1*psz2, ', num2str(n), ', ', num2str(m), '), part=(', num2str(part1), ', ', num2str(part2), '), ', num2str(nm)]);
        end
    end
end
end

function nm = op_compare(O, n, m)
[u, w] = gm_expand(O, n, m);
nm = max(max(abs(O - full_op(u, w, n, m))));
end

function O = full_op(u, w, n, m)
if m > 5
    disp('ERROR - Too many particles for full operator, need m <= 5');
    O = [];
    return
end

d = n^2;
uidx = 1;
O = sparse(n^m, n^m);
for part=1:m
    for mu=1:d
        %if u(uidx) > 0
        [type, alpha, beta] = gm_idx(mu, n);
        Al = sparse(gm_matrix(type, alpha, beta, n));
        A = local_op_to_prod(Al, part, n, m);
        O = O + u(uidx)*A;
        %end
        
        widx = 1;
        for part2=1:m
            for nu=1:d
                if abs(w(uidx, widx)) > 0
                    [type2, alpha2, beta2] = gm_idx(nu, n);
                    Al2 = sparse(gm_matrix(type2, alpha2, beta2, n));
                    A2 = local_op_to_prod(Al2, part2, n, m);
                    O = O + w(uidx, widx)*A*A2;
                end
                widx = widx + 1;
            end
        end
        
        uidx = uidx + 1;
    end
end
end