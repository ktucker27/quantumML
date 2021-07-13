function [E, phi, theta] = povm_entropy(M)

phivec = 0:.01:2*pi;
thetavec = 0:.01:pi;
[theta, phi] = meshgrid(phivec, thetavec);

E = zeros(size(phi));
for ii=1:size(phi,1)
    for jj=1:size(phi,2)
        x = [cos(theta(ii,jj)/2); exp(1i*phi(ii,jj))*sin(theta(ii,jj)/2)];
        rho = x*x';
        p = povm_dist(rho,M);
        if sum(p < 1e-3) > 1
            p
            ii
            jj
        end
        E(ii,jj) = sum(-p.*log(p));
    end
end

if max(max(abs(imag(E)))) > 1e-12
    error('Found complex entropy');
end

E = real(E);