function rho = nqs_density_z(density, a, b, c, w, u, szl, szr)

theta_l = b + w*szl;
theta_r = conj(b) + conj(w)*szr;
theta_m = c + conj(c) + u*szl + conj(u)*szr;

rho = feval(density, a, theta_l, theta_r, theta_m, szl, szr);
