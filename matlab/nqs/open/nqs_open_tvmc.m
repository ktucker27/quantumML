function [t,rho,y] = nqs_open_tvmc(a0, b0, c0, w0, u0, dt, tf)

bb = 0;
jx = 1;
jy = 1;
jz = 1;
gamma = 0;
lloc = @(density, a, b, c, w, u, szl, szr)(afh_lloc(density, a, b, c, w, u, szl, szr, bb, jx, jy, jz, gamma));
n = size(a0,1);
m = size(b0,1);
l = size(c0,1);
num_samps = 1000;
num_steps = 2*n;

odes = @(t,y)(nqs_open_ode(t,y,n,m,l,lloc,num_samps,num_steps));

y0 = [real(a0);imag(a0);...
      real(b0);imag(b0);...
      real(c0);...
      reshape(real(w0),[m*n,1]);reshape(imag(w0),[m*n,1]);...
      reshape(real(u0),[l*n,1]);reshape(imag(u0),[l*n,1])];

if dt > 0
    tvec = (0:dt:tf)';
else
    tvec = [0;tf];
end

options = odeset('RelTol',1e-4,'AbsTol',1e-4);
[t,y] = ode45(odes,tvec,y0,options);

density_op = @(a, theta_l, theta_r, theta_m, szl, szr)(nqs_density(a, theta_l, theta_r, theta_m, szl, szr));

rho = zeros(2^n,2^n,size(t,1));
for i=1:size(t,1)
    sidx = 1;
    a = y(i,sidx:sidx+n-1)' + 1i*y(i,sidx+n:sidx+2*n-1)';
    sidx = sidx + 2*n;
    b = y(i,sidx:sidx+m-1)' + 1i*y(i,sidx+m:sidx+2*m-1)';
    sidx = sidx + 2*m;
    c = y(i,sidx:sidx+l-1)';
    sidx = sidx + l;
    w = reshape(y(i,sidx:sidx+m*n-1)', [m,n]) + 1i*reshape(y(i,sidx+m*n:sidx+2*m*n-1)', [m,n]);
    sidx = sidx + 2*m*n;
    u = reshape(y(i,sidx:sidx+l*n-1)', [l,n]) + 1i*reshape(y(i,sidx+l*n:end)', [l,n]);
    [~,~,rho(:,:,i)] = nqs_density_mat(density_op, a, b, c, w, u);
end