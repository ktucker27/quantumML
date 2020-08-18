function mps_out = dmrg(mpo, mps, tol, maxit)

EPS = 1e-12;

n = mps.num_sites();

% Right normalize the state if it's not already
if ~mps.is_right_normal(EPS)
    mps.right_normalize();
end

ms = mps.substate(1:n);
msd = mps.dagger();

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

for itidx=1:2*maxit
    % Sweep to the right/left updating tensors
    for ii=startidx:idxinc:endidx
        nextidx = ii + idxinc;
        if nextidx > n
            nextidx = n-1;
        elseif nextidx < 1
            nextidx = 2;
        end
        
        % Contract the MPO state with R
        TR = R{ii};
        if ii < n
            TR = TR.end_squeeze();
        end
        A = mpo.tensors{ii}.contract(TR, [2,2]);
        
        % Contract the result with L
        TL = L{ii};
        if ii > 1
            TL = TL.end_squeeze();
        end
        A = A.contract(TL, [1,2]);
        
        % Group the tensor into a matrix and get the eigenvector
        mdims = A.dim([5,3,1]);
        A = A.group({[6,4,2],[5,3,1]});
        [evec, evals] = eig(A.matrix());
        [~, idx] = min(real(diag(evals)));
        evec = evec(:,idx); % TODO - Find a more efficient way of getting this
        M = Tensor(evec);
        if M.rank() ~= 1
            error(['Expected rank one tensor from eigenvector, got ', num2str(M.rank())]);
        end
        M2 = M.split({[1,2,3;mdims]});
        
        % Update the tensors
        if idxinc > 0
            % Left normalize
            M2 = M2.group({[1,3],2});
            [TU, TS, TV] = M2.svd();
            new_m = TU.split({[1,3;mdims([1,3])],2});
            
            % Update the next tensor
            next_m = TS.contract(TV.conjugate(), [2,2]);
            next_m = next_m.contract(ms.tensors{nextidx},[2,1]);
        else
            % Right normalize
            M2 = M2.group({1,[2,3]});
            [TU, TS, TV] = M2.svd();
            new_m = TV.conjugate().split({[2,3;mdims([2,3])],1});
            
            % Update the next tensor
            next_m = TU.contract(TS, [2,1]);
            next_m = ms.tensors{nextidx}.contract(next_m,[2,1]);
            next_m = next_m.split({1,3,2});
        end
        
        ms.set_tensor(ii, new_m);
        msd.set_tensor(ii, new_m.conjugate());
        
        ms.set_tensor(nextidx, next_m);
        msd.set_tensor(nextidx, next_m.conjugate());
        
        if idxinc > 0
            if ii == 1
                % Initialize the L list
                T = mpo.tensors{1}.contract(ms.tensors{1}, [3,3]);
                T = T.contract(msd.tensors{1}, [3,3]);
                L{ii+1} = T.split({5,2,4,1,6,3});
            else
                % Update the L list
                T = L{ii}.contract(ms.tensors{ii}, [1,1]);
                T = T.contract(mpo.tensors{ii}, [1,1;7,3]);
                T = T.contract(msd.tensors{ii}, [1,1;7,3]);
                L{ii+1} = T.split({4,5,6,1,2,3});
            end
        else
            if ii == n
                % Initialize the R list
                T = mpo.tensors{n}.contract(ms.tensors{n}, [3,3]);
                T = T.contract(msd.tensors{n}, [3,3]);
                R{ii-1} = T.split({2,5,1,4,3,6});
            else
                % Update the R list
                T = R{ii}.contract(ms.tensors{ii}, [1,2]);
                T = T.contract(mpo.tensors{ii}, [1,2;7,3]);
                T = T.contract(msd.tensors{ii}, [1,2;7,3]);
                R{ii-1} = T.split({4,5,6,1,2,3});
            end
        end
    end
    
    % Compute energy variance to see if we've converged
    if idxinc < 0
        mpo_ms = apply_mpo(mpo, ms);
        mpo_mpo_ms = apply_mpo(mpo, mpo_ms);
        var = ms.inner(mpo_mpo_ms) - (ms.inner(mpo_ms))^2;
        
        if var < -EPS
            error('Found negative energy variance');
        end
        
        if var < tol
            disp(['Converged in ', num2str(itidx/2), ' iterations with energy variance ', num2str(var)]);
            break;
        end
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
    end
        
    if itidx >= 2*maxit
        disp('WARNING - Max iterations reached');
    end
end

% Build the MPS from the tensor list
mps_out = MPS(ms.tensors);
