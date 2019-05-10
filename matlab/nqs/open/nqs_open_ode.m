function dydt = nqs_open_ode(t, y, n, m, l, lloc, num_samps, num_steps)

global tmax

if t == 0
    tmax = 0;
end

if t > tmax
    disp(['Time: ', num2str(t)]);
    tmax = tmax + .001;
end

sidx = 1;
a = y(sidx:sidx+n-1) + 1i*y(sidx+n:sidx+2*n-1);
sidx = sidx + 2*n;
b = y(sidx:sidx+m-1) + 1i*y(sidx+m:sidx+2*m-1);
sidx = sidx + 2*m;
c = y(sidx:sidx+l-1);
sidx = sidx + l;
w = reshape(y(sidx:sidx+m*n-1), [m,n]) + 1i*reshape(y(sidx+m*n:sidx+2*m*n-1), [m,n]);
sidx = sidx + 2*m*n;
u = reshape(y(sidx:sidx+l*n-1), [l,n]) + 1i*reshape(y(sidx+l*n:end), [l,n]);

[S,F] = nqs_open_sf(a, b, c, w, u, lloc, @nqs_density, num_samps, num_steps);

dydt = minresqlp(S,F);