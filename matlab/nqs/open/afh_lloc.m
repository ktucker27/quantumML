function lloc = afh_lloc(density, a, b, c, w, u, szl, szr, bb, jx, jy, jz, gamma)
% afh_lloc Local Liouvillian for the anisotropic Heisenberg model with
% spontaneous emission

n = size(szl,1);

rho1 = feval(@nqs_density_z, density, a, b, c, w, u, szl, szr);

if abs(rho1) < 1e-10
    lloc = 0;
    return;
end

lloc = 0;
for j=1:n
    % Hamiltonian
    lloc = lloc - 1i*bb*(szl(j) - szr(j));
    
    % Jz
    if j == n
        % Periodic boundary conditions
        lloc = lloc - 1i*jz*(szl(j)*szl(1) - szr(j)*szr(1));
    else
        lloc = lloc - 1i*jz*(szl(j)*szl(j+1) - szr(j)*szr(j+1));
    end
    
    % Jx
    szl_eval = szl;
    szl_eval(j) = -szl(j);
    szr_eval = szr;
    szr_eval(j) = -szr(j);
    if j == n
        % Periodic boundary conditions
        szl_eval(1) = -szl(1);
        szr_eval(1) = -szr(1);
        
        dl = ((szl(j) == -szl(1)) - (szl(j) == szl(1)));
        dr = ((szr(j) == -szr(1)) - (szr(j) == szr(1)));
    else
        szl_eval(j+1) = -szl(j+1);
        szr_eval(j+1) = -szr(j+1);
        
        dl = ((szl(j) == -szl(j+1)) - (szl(j) == szl(j+1)));
        dr = ((szr(j) == -szr(j+1)) - (szr(j) == szr(j+1)));
    end
    
    rho_eval1 = feval(@nqs_density_z, density, a, b, c, w, u, szl_eval, szr);
    rho_eval2 = feval(@nqs_density_z, density, a, b, c, w, u, szl, szr_eval);
    
    lloc = lloc - 1i*jx*(rho_eval1 - rho_eval2)/rho1;
    
    % Jy
    lloc = lloc - 1i*jy*(dl*rho_eval1 - dr*rho_eval2)/rho1;
    
    % Lindblad terms
    if szl(j) == -1 && szr(j) == -1
        szl_eval = szl;
        szl_eval(j) = 1;
        
        szr_eval = szr;
        szr_eval(j) = 1;
        
        rho_num = feval(@nqs_density_z, density, a, b, c, w, u, szl_eval, szr_eval);
        
        %szl_eval(j) = -1;
        %szr_eval(j) = -1;
        
        %rho_denom = feval(density, a, b, c, w, u, szl_eval, szr_eval);
        
        %lloc = lloc + gamma*rho_num/rho_denom;
        % rho_denom should equal rho1
        
        lloc = lloc + gamma*rho_num/rho1;
    end
    
    if szl(j) == 1
        lloc = lloc - gamma/2;
    end
    
    if szr(j) == 1
        lloc = lloc - gamma/2;
    end
end