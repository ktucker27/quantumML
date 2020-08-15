function H = thermal_ham(n, pdim, bq, V)

H = sparse(pdim^n, pdim^n);

for i=1:n
    [spi, smi, szi] = prod_ops(i, pdim, n);
    sxi = 0.5*(spi + smi);
    syi = -0.5*1i*(spi - smi);
    for j=i+1:n
        [spj, smj, szj] = prod_ops(j, pdim, n);
        sxj = 0.5*(spj + smj);
        syj = -0.5*1i*(spj - smj);
        
        H = H + V(i,j)*(szi*szj - 0.5*(sxi*sxj + syi*syj));
    end
    
    H = H + bq*szi*szi;
end