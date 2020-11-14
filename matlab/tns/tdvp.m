function [tvec, mps_out, eout, exp_out] = tdvp(mpo, mps, dt, tfinal, eps, debug, ef, exp_ops)

if nargin < 5
    eps = 0;
end

if nargin < 6
    debug = false;
end

if nargin < 8
    exp_ops = {};
end

TOL = 1e-12;

numt = floor(tfinal/dt + 1);
tvec = zeros(1,numt);
mps_out = cell(1, numt);
eout = zeros(1,numt);
exp_out = zeros(size(exp_ops,2), numt);

n = mps.num_sites();

if imag(dt) == 0
    lanczos_fun = @(x)(exp(1i*x));
    lanczos_mult = 1;
else
    lanczos_fun = @(x)(exp(x));
    lanczos_mult = 1i;
end

% Right normalize the state if it's not already
ms = mps.substate(1:n);
if ~ms.is_right_normal(TOL)
    ms.left_normalize();
    ms.right_normalize();
end

mps_out{1} = MPS(ms.tensors);
msd = ms.dagger();

% Compute the initial energy
mpo_ms = apply_mpo(mpo, ms);
eout(1) = ms.inner(mpo_ms)/ms.inner(ms);
for exp_idx=1:size(exp_ops,2)
    mpo_ms = apply_mpo(exp_ops{exp_idx}, ms);
    exp_out(exp_idx, 1) = ms.inner(mpo_ms)/ms.inner(ms);
end

% If we received a final energy, track the sign of the delta
% so we know when to stop
de = 0;
if nargin > 6 && size(ef,1) ~= 0
    de = sign(eout(1) - ef);
end

% Initialize the R list
R = cell(1,n);
R{n} = Tensor(1,3);
T = mpo.tensors{n}.contract(ms.tensors{n}, [3,3]);
T = T.contract(msd.tensors{n}, [3,3]);
R{n-1} = T.split({2,5,1,4,3,6});
for ii=n-2:-1:1
    T = R{ii+1}.contract(ms.tensors{ii+1}, [1,2]);
    T = T.contract(mpo.tensors{ii+1}, [1,2;7,3]);
    T = T.contract(msd.tensors{ii+1}, [1,2;7,3]);
    R{ii} = T.split({4,5,6,1,2,3});
end

% Initialize the L list
L = cell(1,n);
L{1} = Tensor(1,3);

startidx = 1;
idxinc = 1;
endidx = n;

itidx = 1;
t = 0;
while abs(t) < abs(tfinal)
    % Sweep to the right/left updating tensors
    for ii=startidx:idxinc:endidx
        nextidx = ii + idxinc;
        
        % Get the R tensor
        TR = R{ii};
        if ii < n
            TR = TR.end_squeeze(3);
        end
        
        % Get the L tensor
        TL = L{ii};
        if ii > 1
            TL = TL.end_squeeze(3);
        end
        
        % Evolve according to H
        v = Tensor(ms.tensors{ii}.A, ms.tensors{ii}.rank());
        mdims = v.dims();
        nv = v.norm();
        v.mult_eq(1/nv);
        num_elms = prod(mdims);
        lsteps = min([max([floor(num_elms*0.05),2]), num_elms-1, 10]);
        v = lanczos_expm_mps(TL.mult(-lanczos_mult*dt/2), TR, {mpo.tensors{ii}}, v, lsteps, lanczos_fun)*nv;
        
        M = Tensor(v,1);
        M2 = M.split({[1,2,3;mdims]});
        
        % Update the tensors
        if idxinc > 0
            % Left normalize
            M2 = M2.group({[1,3],2});
            [TU, TS, TV] = M2.svd_trunc(eps);
            new_m = TU.split({[1,3;mdims([1,3])],2});
            
            C = TS.contract(TV.conjugate(), [2,2]);
        else
            % Right normalize
            M2 = M2.group({1,[2,3]});
            [TU, TS, TV] = M2.svd_trunc(eps);
            new_m = TV.conjugate().split({[2,3;mdims([2,3])],1});
            
            C = TU.contract(TS, [2,1]);
        end
        
        ms.set_tensor(ii, new_m, false);
        msd.set_tensor(ii, new_m.conjugate(), false);
        
        % Update the L/R tensor for this site
        if idxinc > 0
            if ii == 1
                % Initialize the L list
                T = mpo.tensors{1}.contract(ms.tensors{1}, [3,3]);
                T = T.contract(msd.tensors{1}, [3,3]);
                L{ii+1} = T.split({5,2,4,1,6,3});
            elseif ii < n
                % Update the L list
                T = L{ii}.contract(ms.tensors{ii}, [1,1]);
                T = T.contract(mpo.tensors{ii}, [1,1;7,3]);
                T = T.contract(msd.tensors{ii}, [1,1;7,3]);
                L{ii+1} = T.split({4,5,6,1,2,3});
            end
            
            if nextidx <= n
                TL = L{nextidx}.end_squeeze(3);
            end
        else
            if ii == n
                % Initialize the R list
                T = mpo.tensors{n}.contract(ms.tensors{n}, [3,3]);
                T = T.contract(msd.tensors{n}, [3,3]);
                R{ii-1} = T.split({2,5,1,4,3,6});
            elseif ii > 1
                % Update the R list
                T = R{ii}.contract(ms.tensors{ii}, [1,2]);
                T = T.contract(mpo.tensors{ii}, [1,2;7,3]);
                T = T.contract(msd.tensors{ii}, [1,2;7,3]);
                R{ii-1} = T.split({4,5,6,1,2,3});
            end
            
            if nextidx >= 1
                TR = R{nextidx}.end_squeeze(3);
            end
        end
        
        if nextidx >= 1 && nextidx <= n
            % Evolve the C tensor backwards in time and contract it into
            % the next two site block
            
            % Evolve according to K
            mdims = C.dims();
            nv = C.norm();
            C.mult_eq(1/nv);
            num_elms = prod(mdims);
            lsteps = min([max([floor(num_elms*0.05),2]), num_elms-1, 10]);
            v = lanczos_expm_mps(TL.mult(lanczos_mult*dt/2), TR, {}, C, lsteps, lanczos_fun)*nv;
            
            C = Tensor(v,1);
            C = C.split({[1,2;mdims]});
            
            % Compute the next site tensor
            if idxinc > 0
                next_m = C.contract(ms.tensors{nextidx},[2,1]);
            else
                next_m = ms.tensors{nextidx}.contract(C,[2,1]);
                next_m = next_m.split({1,3,2});
            end
            
            % Update the next site
            ms.set_tensor(nextidx, next_m, false);
            msd.set_tensor(nextidx, next_m.conjugate(), false);
        end
        
        ms.validate();
        msd.validate();
    end
    
    % Flip the sweep direction
    if idxinc > 0
        startidx = n;
        idxinc = -1;
        endidx = 1;
    else
        startidx = 1;
        idxinc = 1;
        endidx = n;
        
        % Update time and iteration count
        t = t + dt;
        itidx = itidx + 1;
        
        % Update output variables
        tvec(itidx) = t;
        mps_out{itidx} = MPS(ms.tensors);
        
        mpo_ms = apply_mpo(mpo, ms);
        eout(itidx) = ms.inner(mpo_ms)/ms.inner(ms);
        for exp_idx=1:size(exp_ops,2)
            mpo_ms = apply_mpo(exp_ops{exp_idx}, ms);
            exp_out(exp_idx, itidx) = ms.inner(mpo_ms)/ms.inner(ms);
        end
        
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
