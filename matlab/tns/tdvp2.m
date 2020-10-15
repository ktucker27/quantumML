function [tvec, mps_out, eout] = tdvp2(mpo, mps, dt, tfinal, tol, debug, ef)

if nargin < 6
    debug = false;
end

EPS = 1e-12;

numt = floor(tfinal/dt + 1);
tvec = zeros(1,numt);
mps_out = cell(1, numt);
eout = zeros(1,numt);

n = mps.num_sites();

% Right normalize the state if it's not already
ms = mps.substate(1:n);
if ~ms.is_right_normal(EPS)
    ms.left_normalize();
    ms.right_normalize();
end

mps_out{1} = MPS(ms.tensors);
msd = ms.dagger();

% Compute the initial energy
mpo_ms = apply_mpo(mpo, ms);
eout(1) = ms.inner(mpo_ms)/ms.inner(ms);

% If we received a final energy, track the sign of the delta
% so we know when to stop
de = 0;
if nargin > 6
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
endidx = n-1;

itidx = 1;
t = 0;
while abs(t) < abs(tfinal)
    % Sweep to the right/left updating tensors
    for ii=startidx:idxinc:endidx
        nextidx = ii + idxinc;
        
        % Build the two site H matrix
        
        % Contract the MPO state with R
        if idxinc > 0
            ridx = ii + 1;
        else
            ridx = ii;
        end
        TR = R{ridx};
        if ridx < n
            TR = TR.end_squeeze(3);
        end
        H = mpo.tensors{ridx}.contract(TR, [2,2]);
        H = mpo.tensors{ridx-1}.contract(H, [2,1]);
        H = H.group({1,[2,4],[3,5],6,7});
        
        % Contract the result with L
        if idxinc > 0
            lidx = ii;
        else
            lidx = ii-1;
        end
        TL = L{lidx};
        if lidx > 1
            TL = TL.end_squeeze(3);
        end
        H = H.contract(TL, [1,2]);
        
        % Group the tensor into a matrix
        Hmat = H.group({[6,4,2],[5,3,1]});
        
        % Contract and vectorize the current tensors
        v = ms.tensors{ridx-1}.contract(ms.tensors{ridx},[2,1]);
        vdims = v.dim([1,3,2,4]);
        v = v.group({[1,3,2,4]}).A;
        
        % Evolve according to H
        if imag(dt) == 0
            %v = expm(-1i*Hmat.A*dt/2)*v;
            
            fun = @(x)(exp(1i*x));
            nv = norm(v);
            v = lanczos_expm(-Hmat.A*dt/2,v/nv,max([floor(size(v,1)*0.05),2]), fun)*nv;
        else
            nv = norm(v);
            v = lanczos_expm(-1i*Hmat.A*dt/2,v/nv,floor(size(v,1)*0.05))*nv;
        end
        M = Tensor(v);
        
        % Re-arrange the two site tensor as a matrix and perform the SVD
        M2 = M.split({[1,3,2,4;vdims]});
        M2 = M2.group({[1,2],[3,4]});
        [TU, TS, TV] = M2.svd_trunc(tol);
        
        % Update the tensors
        if idxinc > 0
            % Left normalize
            new_m = TU.split({[1,3;vdims([1,3])],2});
            
            C = TS.contract(TV.conjugate(), [2,2]);
            C = C.split({1,[2,3;vdims([2,4])]});
        else
            % Right normalize
            new_m = TV.conjugate().split({[2,3;vdims([2,4])],1});
            
            C = TU.contract(TS, [2,1]);
            C = C.split({[1,3;vdims([1,3])],2});
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
        
        if nextidx >= 2 && nextidx <= n-1
            % Evolve the second site backwards in time and contract it into
            % the next two site block
            
            % Build the one site H matrix
            H = mpo.tensors{nextidx}.contract(TR, [2,2]);
            H = H.contract(TL, [1,2]);
            
            % Group the tensor into a matrix
            mdims = H.dim([5,3,1]);
            Hmat = H.group({[6,4,2],[5,3,1]});
            
            % Vectorize the next tensor
            v = C.group({[1,2,3]}).A;
            
            % Evolve according to H
            if imag(dt) == 0
                %v = expm(1i*Hmat.A*dt/2)*v;
                
                fun = @(x)(exp(1i*x));
                nv = norm(v);
                v = lanczos_expm(Hmat.A*dt/2,v/nv,max([floor(size(v,1)*0.05),2]), fun)*nv;
            else
                nv = norm(v);
                v = lanczos_expm(1i*Hmat.A*dt/2,v/nv,floor(size(v,1)*0.05))*nv;
            end
            C = Tensor(v);
            next_m = C.split({[1,2,3;mdims]});
        else
            % The next site is an end site, so normalize the C tensor as 
            % appropriate and put it in the MPS. For an end site, this 
            % throws away a scalar, effectively normalizing the MPS
            
            cdims = C.dims();
            if idxinc > 0
                % Left normalize
                C = C.group({[1,3],2});
                [TU, ~, ~] = C.svd();
                next_m = TU.split({[1,3;cdims([1,3])],2});
            else
                % Right normalize
                C = C.group({1,[2,3]});
                [~, ~, TV] = C.svd();
                next_m = TV.conjugate().split({[2,3;cdims([2,3])],1});
            end
        end
        
        % Update the next site
        ms.set_tensor(nextidx, next_m, false);
        msd.set_tensor(nextidx, next_m.conjugate(), false);
        
        ms.validate();
        msd.validate();
    end
    
    % Flip the sweep direction
    if idxinc > 0
        startidx = n;
        idxinc = -1;
        endidx = 2;
    else
        startidx = 1;
        idxinc = 1;
        endidx = n-1;
        
        % Update time and iteration count
        t = t + dt;
        itidx = itidx + 1;
        
        % Update output variables
        tvec(itidx) = t;
        mps_out{itidx} = MPS(ms.tensors);
        
        mpo_ms = apply_mpo(mpo, ms);
        eout(itidx) = ms.inner(mpo_ms)/ms.inner(ms);
        
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
