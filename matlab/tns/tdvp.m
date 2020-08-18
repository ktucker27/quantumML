function [tvec, mps_out, eout] = tdvp(mpo, mps, dt, tfinal, debug)

if nargin < 5
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
        
        % Build the H matrix
        
        % Contract the MPO state with R
        TR = R{ii};
        if ii < n
            TR = TR.end_squeeze();
        end
        H = mpo.tensors{ii}.contract(TR, [2,2]);
        
        % Contract the result with L
        TL = L{ii};
        if ii > 1
            TL = TL.end_squeeze();
        end
        H = H.contract(TL, [1,2]);
        
        % Group the tensor into a matrix
        mdims = H.dim([5,3,1]);
        H = H.group({[6,4,2],[5,3,1]});
        
        % Vectorize the current tensor
        v = ms.tensors{ii}.group({[1,2,3]}).A;
        
        % Evolve according to H
        v = expm(-1i*H.A*dt/2)*v;
        M = Tensor(v);
        M2 = M.split({[1,2,3;mdims]});
        
        % Update the tensors
        if idxinc > 0
            % Left normalize
            M2 = M2.group({[1,3],2});
            [TU, TS, TV] = M2.svd();
            new_m = TU.split({[1,3;mdims([1,3])],2});
            
            C = TS.contract(TV.conjugate(), [2,2]);
        else
            % Right normalize
            M2 = M2.group({1,[2,3]});
            [TU, TS, TV] = M2.svd();
            new_m = TV.conjugate().split({[2,3;mdims([2,3])],1});
            
            C = TU.contract(TS, [2,1]);
        end
        
        ms.set_tensor(ii, new_m);
        msd.set_tensor(ii, new_m.conjugate());
        
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
                TL = L{nextidx}.end_squeeze();
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
                TR = R{nextidx}.end_squeeze();
            end
        end
        
        if nextidx >= 1 && nextidx <= n
            % Build the K matrix
            K = TL.contract(TR, [2,2]);
            
            % Group the tensor into a matrix
            mdims = K.dim([1,3]);
            K = K.group({[2,4],[1,3]});
            
            % Vectorize the next tensor
            v = C.group({[1,2]}).A;
            
            % Evolve according to K
            v = expm(1i*K.A*dt/2)*v;
            C = Tensor(v);
            C = C.split({[1,2;mdims]});
            
            % Compute the next site tensor
            if idxinc > 0
                next_m = C.contract(ms.tensors{nextidx},[2,1]);
            else
                next_m = ms.tensors{nextidx}.contract(C,[2,1]);
                next_m = next_m.split({1,3,2});
            end
            
            % Update the next site
            ms.set_tensor(nextidx, next_m);
            msd.set_tensor(nextidx, next_m.conjugate());
        end
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
        
        if debug
            if mod(itidx-1,10) == 0
                disp(['t = ', num2str(t)]);
            end
        end
    end
end
