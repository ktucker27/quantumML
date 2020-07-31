function H = build_ham(one_site, two_site, pdim, n)

H = zeros(pdim^n, pdim^n);
for ii=1:n
    for jj=1:size(one_site,2)
        H = H + local_op_to_prod(one_site{jj}, ii, pdim, n);
    end
    
    if ii < n
        for jj=1:size(two_site,2)
            A = local_op_to_prod(two_site{jj}{1}, ii, pdim, n);
            B = local_op_to_prod(two_site{jj}{2}, ii+1, pdim, n);
            H = H + A*B;
        end
    end
end