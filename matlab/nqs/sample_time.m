function psimat = sample_time(psi0, H, dt, tf, num_samp, outdir, prefix)

psimat = zeros(size(psi0,1), round(tf/dt) + 1);
psi = psi0;
dtmat = expm(-1i*H*dt);
psiidx = 1;
for t=0:dt:tf
    psimat(:,psiidx) = psi;
    psiidx = psiidx + 1;
    
    sz = cdf_sample(psi, num_samp);
    if nargin > 5
        dlmwrite([outdir, '/', prefix, '_t', strrep(num2str(t, 2), '.', 'p'), '.txt'], sz, 'delimiter', ' ');
    end
    psi = dtmat*psi;
end