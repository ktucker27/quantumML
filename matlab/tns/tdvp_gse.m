function [tvec, mps_out, eout] = tdvp_gse(mpo, mpo_exp, kdim, mps, dt, tfinal, eps, debug, ef)

epsk = eps(1);
epsm = eps(2);
eps_tdvp = eps(3);

if nargin < 8
    debug = false;
end

TOL = 1e-12;

numt = floor(tfinal/dt + 1);
tvec = zeros(1,numt);
mps_out = cell(1, numt);
mps_exp = cell(1, kdim);
eout = zeros(1,numt);

n = mps.num_sites();

% Left normalize the state if it's not already
ms = mps.substate(1:n);
if ~ms.is_right_normal(TOL)
    ms.right_normalize();
    ms.left_normalize();
end

mps_out{1} = MPS(ms.tensors);

% Calculate the max bond dimensions
max_r = zeros(1,n-1);
leftmax = 1;
rightmax = 1;
for ii=1:floor(n/2)
    leftmax = leftmax*ms.tensors{ii}.dim(3);
    rightmax = rightmax*ms.tensors{n-ii+1}.dim(3);
    max_r(ii) = leftmax;
    max_r(n-ii) = rightmax;
end

% Compute the initial energy
mpo_ms = apply_mpo(mpo, ms);
eout(1) = ms.inner(mpo_ms)/ms.inner(ms);

% If we received a final energy, track the sign of the delta
% so we know when to stop
de = 0;
if nargin > 8
    de = sign(eout(1) - ef);
end

itidx = 1;
t = 0;
while abs(t) < abs(tfinal)
    % Compute Krylov subspace states
    mps_exp{1} = ms;
    for kk=2:kdim
        mps_exp{kk} = apply_mpo(mpo_exp, mps_exp{kk-1});
        mps_exp{kk}.left_normalize(epsk);
    end
    
    % Use the subspace vectors to do a basis extension
    for site_idx=n:-1:1
        % Perform the SVD of the output state without truncation
        if site_idx == n
            C = ms.tensors{site_idx};
        end
        
        cdims = C.dims();
        TCmat = C.group({1,[2,3]});
        
        [~, ~, TV] = TCmat.svd();
        
        % Compute the projector onto the null space
        B = TV.A';
        
        if site_idx > 1 && size(B,1) < cdims(2)*cdims(3) && size(B,1) < max_r(site_idx-1)
            cols = size(B,2);
            P = eye(cols) - B'*B;
            
            % Compute the combined density operator for all the other states
            rho = zeros(cols, cols);
            for kk=2:kdim
                C_tilde = mps_exp{kk}.tensors{site_idx};
                
                C_tilde_mat = C_tilde.group({1,[2,3]}).A;
                
                rho = rho + C_tilde_mat'*C_tilde_mat;
            end
            
            % Project onto the nullspace if it has a nontrivial dimension
            if norm(P)/norm(rho) > TOL
                trace_rho = trace(rho);
                rho = P*rho*P;
                
                if(trace(rho)/trace_rho > 1e-10)
                    % Diagonalize and truncate rho
                    [~, Sbar, Bbar] = svd(rho);
                    end_idx = get_trunc_idx(diag(Sbar), epsm);
                    
                    % Extract the new basis
                    Bbar = Bbar(:,1:end_idx);
                    
                    B = [B;Bbar']; %#ok<AGROW>
                end
            end
        end
        
        TB = Tensor(B);
        
        TB = TB.split({1,[2,3;cdims([2,3])]});
        ms.set_tensor(site_idx, TB, false);
        
        % Update the tensors at the next site
        if site_idx > 1
            CBdag = C.contract(TB.conjugate(),[2,2;3,3]);
            C = ms.tensors{site_idx-1}.contract(CBdag, [2,1]);
            C = C.split({1,3,2});
            
            for kk=2:kdim
                C_tilde_Bdag = mps_exp{kk}.tensors{site_idx}.contract(TB.conjugate(),[2,2;3,3]);
                C_tilde = mps_exp{kk}.tensors{site_idx-1}.contract(C_tilde_Bdag, [2,1]);
                C_tilde = C_tilde.split({1,3,2});
                mps_exp{kk}.set_tensor(site_idx-1, C_tilde, false);
            end
        end
    end
    ms.validate();
    
    % Perform a single TDVP pass
    [~, tdvp_mps_out, tdvp_eout] = tdvp(mpo, ms, dt, dt, eps_tdvp, false);
    ms = tdvp_mps_out{end};
    eout(itidx) = tdvp_eout(end);
    
    % Make sure we only did one iteration of TDVP
    if size(tdvp_eout,2) ~= 2
        error(['Expected to do a single TDVP step, got ', num2str(size(eout,2))]);
    end
    
    % Update time and iteration count
    t = t + dt;
    itidx = itidx + 1;
    
    % Update output variables
    tvec(itidx) = t;
    mps_out{itidx} = MPS(ms.tensors);
    
    if de ~= 0
        if de ~= sign(eout(itidx) - ef)
            eout = eout(1:itidx);
            tvec = tvec(1:itidx);
            mps_out = mps_out(1:itidx);
            break
        end
    end
    
    if debug
        if mod(itidx-1,10) == 0
            disp(['t = ', num2str(t)]);
        end
    end
end

end
